def build_context(chunks):
    """
    Строит контекст из полученных чанков
    """
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"[DOC {i+1}]\n"
        context += f"Title: {chunk['metadata']['title']}\n"
        context += f"URL: {chunk['metadata']['url']}\n"
        context += f"Content:\n {chunk['content']}\n\n"
    return context