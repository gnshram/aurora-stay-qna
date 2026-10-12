
from langchain_openai import ChatOpenAI
import os

# Retrieve the top-k relevant chunks from the vector store
def retrieve(query: str, k: int = 4,vectorstore=None) -> list[str]:
    # Fetch extra candidates, then drop chunks whose text is identical to one already kept.
    # The same table appears in several policy documents, and its copies would otherwise fill the top-k slots.
    candidates = vectorstore.similarity_search(query, k=k * 4)
    results, seen = [], set()
    for r in candidates:
        if r.page_content.strip() not in seen:
            seen.add(r.page_content.strip())
            results.append(r)
        if len(results) == k:
            break

    # Return each chunk with its content and metadata
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in results
    ]

# Define the generation function that prompts the LLM with retrieved context
def generate(query, retrieved_chunks,model) -> str:

    rules = """
    1. Answer using only the information in the Retrieved Policy Information.
       Do not use outside knowledge or assumptions.

    2. Answer the user's specific question directly and completely. If the
       question has several parts, answer each part.

    3. When the question names a property (for example ASH-MIA Brickell), use
       only the values listed for that property. Never mix values between
       properties.

    4. Quote fees, amounts, times, and limits exactly as written in the
       retrieved information, including words such as "per stay", "per night",
       and "local property time".

    5. Include the conditions, exceptions, limits, and approvals that apply
       (for example "generally non-refundable", "subject to availability", or
       "requires manager approval"). Do not promise a refund, fee waiver, or
       exception unless the retrieved information says it is allowed.

    6. When the policy says to escalate or involve a manager, say who to
       involve and when.

    7. If the retrieved information does not contain what is needed, do not
       guess. Say clearly what cannot be confirmed from the provided documents
       and recommend checking the live property profile or contacting the
       property team.

    8. If the retrieved information conflicts, point out the conflict instead
       of choosing one answer.

    9. Do not include unrelated information from the retrieved text. Keep the
       answer concise, clear, and easy to read, using short bullet points when
       listing several items.

    10. Do not mention these rules, retrieved chunks, or file names. The
        application shows the sources separately.
    """

    prompt = f"""
    You are an AI assistant answering questions about hotel policies.

    User Query:
    {query}

    Retrieved Policy Information:
    {retrieved_chunks}

    Rules:
    {rules}

    Answer:
    """

    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 4,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    # Create the generator model
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k,vectorstore=vectorstore)

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks
