import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    backends: Dict[str, Dict[str, str]] = {}
    current_dir = Path(".")

    # TODO: Create list of directories that match specific criteria (directory type and name pattern)
    chroma_dirs = [
        d for d in current_dir.iterdir()
        if d.is_dir() and "chroma" in d.name.lower()
    ]

    # TODO: Loop through each discovered directory
    for chroma_dir in chroma_dirs:
        # TODO: Wrap connection attempt in try-except block for error handling
        try:
            # TODO: Initialize database client with directory path and configuration settings
            client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=Settings(anonymized_telemetry=False)
            )

            # TODO: Retrieve list of available collections from the database
            collections = client.list_collections()

            # TODO: Loop through each collection found
            for col in collections:
                # TODO: Create unique identifier key combining directory and collection names
                key = f"{chroma_dir.name}:{col.name}"

                # TODO: Get document count with fallback for unsupported operations
                try:
                    count = col.count()
                except Exception:
                    count = "unknown"

                # TODO: Build information dictionary
                backends[key] = {
                    # TODO: Store directory path as string
                    "directory": str(chroma_dir),
                    # TODO: Store collection name
                    "collection_name": col.name,
                    # TODO: Create user-friendly display name
                    "display_name": f"{chroma_dir.name} / {col.name} ({count} docs)",
                    "count": count,
                }

        # TODO: Handle connection or access errors gracefully
        except Exception as e:
            key = f"{chroma_dir.name}:error"
            backends[key] = {
                # TODO: Create fallback entry for inaccessible directories
                "path": str(chroma_dir),
                "collection": "N/A",
                # TODO: Include error information in display name with truncation
                "display": f"{chroma_dir.name} (error: {str(e)[:40]}...)",
                # TODO: Set appropriate fallback values for missing information
                "count": 0,
            }

    # TODO: Return complete backends dictionary with all discovered collections
    return backends


def initialize_rag_system(chroma_dir: str, collection_name: str):
    # TODO: Create a chomadb persistentclient
 
    # TODO: Return the collection with the collection_name
    try:
        # Create ChromaDB persistent client
        client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection
        collection = client.get_or_create_collection(collection_name)

        return collection, True, None

    except Exception as e:
        return None, False, str(e)


def retrieve_documents(
    collection,
    query: str,
    n_results: int = 3,
    mission_filter: Optional[str] = None
) -> Optional[Dict]:

    # TODO: Initialize filter variable to None (represents no filtering)
    where = None

    # TODO: Check if filter parameter exists and is not set to "all" or equivalent
    if mission_filter and mission_filter.lower() != "all":
        # TODO: If filter conditions are met, create filter dictionary
        where = {"mission": mission_filter}

    # TODO: Execute database query
    results = collection.query(
        # TODO: Pass search query in the required format
        query_texts=[query],
        # TODO: Set maximum number of results to return
        n_results=n_results,
        # TODO: Apply conditional filter
        where=where
    )

    # TODO: Return query results to caller
    return results


def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    if not documents:
        return ""

    # TODO: Initialize list with header text for context section
    context_parts = ["### Retrieved Context\n"]

    # TODO: Loop through paired documents and their metadata using enumeration
    for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        # TODO: Extract mission information from metadata with fallback value
        mission = meta.get("mission", "unknown").replace("_", " ").title()
        # TODO: Extract source information from metadata with fallback value
        source = meta.get("source", "unknown")
        # TODO: Extract category information from metadata with fallback value
        category = meta.get("category", "unknown").replace("_", " ").title()

        # TODO: Create formatted source header with index number and extracted information
        header = f"Source {i} | Mission: {mission} | Category: {category} | File: {source}"
        # TODO: Add source header to context parts list
        context_parts.append(header)

        # TODO: Check document length and truncate if necessary
        if len(doc) > 1000:
            doc = doc[:1000] + "..."

        # TODO: Add truncated or full document content to context parts list
        context_parts.append(doc)
        context_parts.append("")

    # TODO: Join all context parts with newlines and return formatted string
    return "\n".join(context_parts)
