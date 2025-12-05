import json
import re
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import List, Sequence
from uuid import uuid4

import regex
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_text_splitters.markdown import ExperimentalMarkdownSyntaxTextSplitter
from overrides import override
from tqdm import tqdm

_TABLE_PATTERN = re.compile(r"\[\[TABLE_\d+\]\]")


class MarkdownSplitter(ExperimentalMarkdownSyntaxTextSplitter):
    """
    Enhanced Markdown splitter that handles tables and code blocks specifically.
    """

    def __init__(
        self,
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
        separators=None,
        multi_process=False,
        show_progress=False,
        include_metadata: Sequence[str] = (),
        *args,
        **kwargs,
    ):
        self.postprocessing_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            is_separator_regex=is_separator_regex,
            separators=separators,
        )
        self.chunks_overlap = chunk_overlap
        self._multi_process = multi_process
        self._show_progress = show_progress
        self._metadata_header_fields = include_metadata
        super().__init__(*args, **kwargs)

    def redefine_postprocessing_splitter(
        self, splitter: TextSplitter, *args, **kwargs
    ) -> None:
        self.postprocessing_splitter = splitter(*args, **kwargs)

    def process_markdown_tables(self, text: str, keep: bool = False) -> str:
        """Removes markdown tables."""
        pattern = re.compile(r"^(?:\s*\|.*\|\s*(?:\n|$))+", re.MULTILINE)
        return pattern.sub("", text)

    def _placehold_tables(self, text: str, table_chunk_size: int = 3):
        """Replaces tables with placeholders [[TABLE_x]]."""
        self._table_chunks_map = {}
        lines = text.splitlines()
        output_lines = []
        table_index = 0
        i = 0
        while i < len(lines):
            if (
                i + 1 < len(lines)
                and "|" in lines[i]
                and re.match(r"^(?:\s*\|.*\|\s*(?:\n|$))+", lines[i + 1])
            ):
                table_start = i
                j = i + 2
                while (
                    j < len(lines) and lines[j].strip() != "" and "|" in lines[j]
                ):
                    j += 1
                table_end = j - 1

                table_block = "\n".join(lines[table_start : table_end + 1])
                table_chunks = self._chunk_table(table_block, table_chunk_size)

                placeholder = f"[[TABLE_{table_index}]]"

                self._table_chunks_map[placeholder] = table_chunks
                if output_lines and output_lines[-1].strip() != "":
                    output_lines.append("")
                output_lines.append(placeholder)
                if j < len(lines) and lines[j].strip() != "":
                    output_lines.append("")
                i = j
                table_index += 1
                continue
            output_lines.append(lines[i])
            i += 1

        return "\n".join(output_lines)

    def _chunk_table(self, table_text: str, chunk_size: int = 3) -> List[str]:
        lines = table_text.splitlines()
        if not lines:
            return []
        data_lines = lines[2:] if len(lines) > 2 else []
        chunks = []
        if data_lines:
            for idx in range(0, len(data_lines), chunk_size):
                group = data_lines[idx : idx + chunk_size]
                chunks.append("\n".join(group))
        return chunks

    def remove_xml_blocks(self, text: str) -> str:
        pattern_1 = regex.compile(
            r"""(?sx)                             
            ^<\?xml\s+version="1\.0"\s+encoding="utf-8"\?>   
            (?:
                (?!                             
                    \n                      
                    (?:                      
                        (?:(?<!<[^>\n]*)\s*[^<])
                    )
                )
                .| \n    
            )+
            """,
            regex.MULTILINE,
            regex.IGNORECASE,
        )
        pattern_2 = re.compile(r"```\s*(?:xml)?\n?<\?xml\s+version=.*\n*```\s*")
        pattern_3 = re.compile(r"```xml\s*.*?```", re.DOTALL)
        temp = pattern_1.sub("", text)
        temp = pattern_2.sub("", temp)
        return pattern_3.sub("", temp)

    def process_json_arrays(self, data):
        """Simplifies JSON arrays to keep only the first element recursively."""
        if isinstance(data, list):
            if len(data) > 0:
                processed = self.process_json_arrays(data[0])
                if processed is None:
                    return None
                return [processed]
            else:
                return None
        elif isinstance(data, dict):
            new_data = {}
            for key, value in data.items():
                processed_value = self.process_json_arrays(value)
                if processed_value is not None:
                    new_data[key] = processed_value
            return new_data
        else:
            return data

    def process_code_blocks(self, text: str) -> str:
        """Cleans SQL and simplifies JSON code blocks."""
        candidate_pattern = re.compile(
            r"```\s*\n?(?:\s*(?:sql)?(?:true)?)(.*?)\n?```",
            flags=re.DOTALL | re.IGNORECASE,
        )

        def repl(match):
            candidate_block = match.group(0)
            candidate_code = match.group(1).strip()
            try:
                data = json.loads(candidate_code)
            except json.JSONDecodeError:
                return candidate_block

            processed_data = self.process_json_arrays(data)
            if processed_data is None:
                return ""
            new_code = json.dumps(processed_data, ensure_ascii=False)
            return "```\n" + new_code + "\n```"

        return candidate_pattern.sub(repl, text)

    def remove_md_links(self, text: str) -> str:
        pattern = re.compile(r"\[(.*?)\]\(.*?\)")
        return pattern.sub(r"\1", text)

    def _preprocess_text(
        self,
        text: str,
        keep_tables: bool,
        keep_links: bool = False,
        table_chunk_size: int = 3,
    ) -> str:
        if not keep_links:
            text = self.remove_md_links(text)
        text = self.process_code_blocks(text)
        if not keep_tables:
            text = self.process_markdown_tables(text)
        else:
            text = self._placehold_tables(text, table_chunk_size)
        return text

    def _postprocess_documents(
        self,
        documents: List[Document],
        initial_document_metadata: dict | None = None,
        max_symbols: int = 1500,
        keep_tables: bool = False,
    ) -> List[Document]:
        seen_contents = set()
        postprocessed_documents = []

        header_parts = []
        if initial_document_metadata:
            for field_name in self._metadata_header_fields:
                value = initial_document_metadata.get(field_name)
                if value:
                    header_parts.append(f"{field_name}: {str(value)}")
        header_base = "\n\n".join(header_parts)
        if header_base:
            header_base = f"{header_base}\n\n"

        has_table_map = hasattr(self, "_table_chunks_map")

        for chunk in documents:
            if chunk.metadata and "Code" in chunk.metadata:
                del chunk.metadata["Code"]
            headers = f"{chunk.metadata}" if chunk.metadata else ""
            header_info_string = (
                (f"{header_base}{headers}\n\n" if headers else header_base)
                if header_base or headers
                else ""
            )
            text = chunk.page_content

            needs_table_processing = (
                keep_tables and "[[TABLE_" in text and has_table_map
            )

            if needs_table_processing:
                self._process_chunk_with_tables(
                    text,
                    header_info_string,
                    initial_document_metadata,
                    max_symbols,
                    postprocessed_documents,
                    seen_contents,
                )
            else:
                self._process_plain_chunk(
                    text,
                    header_info_string,
                    initial_document_metadata,
                    max_symbols,
                    postprocessed_documents,
                    seen_contents,
                )

        return postprocessed_documents

    def _process_chunk_with_tables(
        self,
        text: str,
        header: str,
        metadata: dict | None,
        max_symbols: int,
        result_docs: List[Document],
        seen_contents: set,
    ) -> None:
        parts = _TABLE_PATTERN.split(text)
        table_markers = _TABLE_PATTERN.findall(text)

        for i, part in enumerate(parts):
            if part.strip():
                self._add_text_part(
                    part, header, metadata, max_symbols, result_docs, seen_contents
                )

            if i < len(table_markers):
                self._add_table_chunks(
                    table_markers[i],
                    header,
                    metadata,
                    max_symbols,
                    result_docs,
                    seen_contents,
                )

    def _process_plain_chunk(
        self,
        text: str,
        header: str,
        metadata: dict | None,
        max_symbols: int,
        result_docs: List[Document],
        seen_contents: set,
    ) -> None:
        self._add_text_part(
            text, header, metadata, max_symbols, result_docs, seen_contents
        )

    def _add_text_part(
        self,
        text: str,
        header: str,
        metadata: dict | None,
        max_symbols: int,
        result_docs: List[Document],
        seen_contents: set,
    ) -> None:
        initial_text = text

        if len(text) > max_symbols:
            sub_chunks = self.postprocessing_splitter.split_text(text)
            for sub_chunk in sub_chunks:
                if len(sub_chunk) < self.chunks_overlap:
                    continue
                full_content = f"{header}{sub_chunk}"
                if full_content not in seen_contents:
                    seen_contents.add(full_content)
                    result_docs.append(
                        Document(
                            full_content,
                            metadata={
                                **(metadata or {}),
                                "initial_text": initial_text,
                                "chunk_id": str(uuid4()),
                            },
                        )
                    )
        else:
            full_content = f"{header}{text}"
            if full_content not in seen_contents:
                seen_contents.add(full_content)
                result_docs.append(
                    Document(
                        full_content,
                        metadata={
                            **(metadata or {}),
                            "initial_text": initial_text,
                            "chunk_id": str(uuid4()),
                        },
                    )
                )

    def _add_table_chunks(
        self,
        table_marker: str,
        header: str,
        metadata: dict | None,
        max_symbols: int,
        result_docs: List[Document],
        seen_contents: set,
    ) -> None:
        table_chunks = self._table_chunks_map.get(table_marker, [])
        for table_chunk in table_chunks:
            initial_text = table_chunk

            if len(table_chunk) > max_symbols:
                sub_chunks = self.postprocessing_splitter.split_text(table_chunk)
                for sub_chunk in sub_chunks:
                    if len(sub_chunk) < self.chunks_overlap:
                        continue
                    full_content = f"{header}{sub_chunk}"
                    if full_content not in seen_contents:
                        seen_contents.add(full_content)
                        result_docs.append(
                            Document(
                                full_content,
                                metadata={
                                    **(metadata or {}),
                                    "initial_text": initial_text,
                                    "chunk_id": str(uuid4()),
                                },
                            )
                        )
            else:
                full_content = f"{header}{table_chunk}"
                if full_content not in seen_contents:
                    seen_contents.add(full_content)
                    result_docs.append(
                        Document(
                            full_content,
                            metadata={
                                **(metadata or {}),
                                "initial_text": initial_text,
                                "chunk_id": str(uuid4()),
                            },
                        )
                    )

    @override
    def split_text(
        self,
        text: str,
        document_metadata: dict = None,
        keep_tables: bool = False,
        keep_links: bool = False,
        max_symbols: int = 1500,
        table_chunk_size: int = 3,
    ) -> List[Document]:
        text = self._preprocess_text(
            text,
            keep_tables=keep_tables,
            keep_links=keep_links,
            table_chunk_size=table_chunk_size,
        )
        splitted_text = super().split_text(text)

        postprocessed_documents = self._postprocess_documents(
            splitted_text, document_metadata, max_symbols, keep_tables=keep_tables
        )
        return postprocessed_documents

    def split_documents(
        self,
        documents: List[Document],
        keep_tables: bool = False,
        keep_links: bool = False,
        max_symbols: int = 1500,
        table_chunk_size: int = 3,
    ) -> List[Document]:
        split_progress_tracker = partial(
            tqdm,
            total=len(documents),
            desc="Splitting documents",
            disable=not self._show_progress,
        )

        if self._multi_process:
            with ProcessPoolExecutor() as executor:
                split_document_chunks = list(
                    split_progress_tracker(
                        iterable=executor.map(
                            self.process_document,
                            documents,
                            [keep_tables] * len(documents),
                            [keep_links] * len(documents),
                            [max_symbols] * len(documents),
                            [table_chunk_size] * len(documents),
                        ),
                    )
                )
        else:
            split_document_chunks = list(
                split_progress_tracker(
                    iterable=(
                        self.process_document(
                            doc,
                            keep_tables=keep_tables,
                            keep_links=keep_links,
                            max_symbols=max_symbols,
                            table_chunk_size=table_chunk_size,
                        )
                        for doc in documents
                    ),
                )
            )

        result = []
        for splitted_document in split_document_chunks:
            result.extend(splitted_document)

        return result

    def process_document(
        self, doc, keep_tables, keep_links, max_symbols, table_chunk_size
    ):
        return self.split_text(
            doc.page_content,
            document_metadata=doc.metadata,
            keep_tables=keep_tables,
            keep_links=keep_links,
            max_symbols=max_symbols,
            table_chunk_size=table_chunk_size,
        )
