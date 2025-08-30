from pinecone.grpc import PineconeGRPC, GRPCClientConfig
import os


def get_local_index(index_name: str, host: str = None, secure: bool = False):
    """
    Connect to Pinecone Local and return a gRPC Index handle.
    Raises a clear error if the index doesn't exist on that host.
    """
    host = host or os.getenv("PINECONE_LOCAL_HOST", "http://localhost:5080")
    pc = PineconeGRPC(api_key=os.getenv("PINECONE_API_KEY", "pclocal"), host=host)

    # Check index exists on this host
    if not pc.has_index(index_name):
        raise RuntimeError(
            f"Index '{index_name}' not found on Pinecone Local at {host}. "
            f"Make sure your embedder created it on this host with exactly the same name."
        )

    idx_host = pc.describe_index(name=index_name).host
    # secure=False for local, no TLS
    return pc.Index(host=idx_host, grpc_config=GRPCClientConfig(secure=secure))


def debug_namespaces_local(index) -> list[str]:
    """Return and print namespaces present on the local index."""
    stats = index.describe_index_stats() or {}
    ns_map = stats.get("namespaces") or {}
    namespaces = list(ns_map.keys())
    # include default probe
    if "" not in namespaces:
        namespaces.append("")
    print(f"[DEBUG] Local index namespaces: {namespaces}")
    return namespaces

if __name__ == '__main__':
    index = get_local_index("legal-clauses", secure=False)
    debug_namespaces_local(index)