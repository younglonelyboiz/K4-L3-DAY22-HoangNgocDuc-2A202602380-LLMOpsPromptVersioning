"""
Tiện ích để tải và xử lý dữ liệu cho RAG pipeline.

Cách dùng:
    from utils.data_loader import load_knowledge_base, split_text, build_vectorstore

    text        = load_knowledge_base()
    chunks      = split_text(text, chunk_size=500, chunk_overlap=50)
    vectorstore = build_vectorstore(chunks, embeddings)
"""
from pathlib import Path


def load_knowledge_base(path: str = None) -> str:
    """
    Đọc file knowledge base và trả về nội dung dạng chuỗi.

    Args:
        path: đường dẫn tới file text.
              Mặc định: data/knowledge_base.txt (thư mục gốc của project)

    Returns:
        Nội dung file dưới dạng str
    """
    if path is None:
        path = Path(__file__).parent.parent.parent / "data" / "knowledge_base.txt"
    return Path(path).read_text(encoding="utf-8")


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """
    Chia văn bản thành các đoạn nhỏ (chunks) để index.

    Dùng RecursiveCharacterTextSplitter — tách ưu tiên theo đoạn văn, câu, rồi ký tự.

    Args:
        text         : văn bản cần chia
        chunk_size   : số ký tự tối đa mỗi chunk (mặc định: 500)
        chunk_overlap: số ký tự chồng lên nhau giữa 2 chunks liên tiếp (mặc định: 50)

    Returns:
        list[str] — danh sách các chuỗi chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


def build_vectorstore(chunks: list, embeddings):
    """
    Tạo FAISS vectorstore từ danh sách chunks và embeddings.
    Tự động lưu và tải từ cache (data/faiss_index) để tránh tốn quota embedding nhiều lần.

    Args:
        chunks    : list[str] — danh sách text chunks đã chia
        embeddings: Embeddings instance (từ get_embeddings())

    Returns:
        FAISS vectorstore đã được index và sẵn sàng dùng để retrieve
    """
    import os
    import time
    from langchain_community.vectorstores import FAISS

    cache_dir = Path(__file__).parent.parent.parent / "data" / "faiss_index"

    # 1. Kiểm tra cache nếu đã tồn tại
    if cache_dir.exists():
        print(f"📦 Đang tải FAISS vectorstore từ cache ({cache_dir}) ...")
        try:
            vectorstore = FAISS.load_local(str(cache_dir), embeddings, allow_dangerous_deserialization=True)
            print("✅ Đã tải FAISS vectorstore từ cache thành công! (Không tốn quota)")
            return vectorstore
        except Exception as e:
            print(f"⚠️ Không tải được cache: {e}. Đang tạo lại...")

    try:
        import config
        is_local = getattr(config, "PROVIDER", "").lower() == "ollama"
    except Exception:
        is_local = False

    BATCH_SIZE = 50 if is_local else 5  # local batch 50 để chạy nhanh, cloud batch 5 tránh rate limit
    MAX_RETRIES = 3

    print(f"🔨 Đang tạo FAISS index từ {len(chunks)} chunks ...")

    vectorstore = None
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"   ⏳ Batch {batch_num}/{total_batches} ({len(batch)} chunks) ...")

        for attempt in range(MAX_RETRIES):
            try:
                if vectorstore is None:
                    vectorstore = FAISS.from_texts(batch, embeddings)
                else:
                    vectorstore.add_texts(batch)
                break  # thành công → thoát retry loop
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = 10 * (attempt + 1)  # 10s, 20s, 30s
                    print(f"   ⚠️  Rate limit, chờ {wait}s rồi thử lại ({attempt + 1}/{MAX_RETRIES})...")
                    time.sleep(wait)
                else:
                    raise  # lỗi khác → raise ngay
        else:
            raise RuntimeError(f"Batch {batch_num} thất bại sau {MAX_RETRIES} lần thử")

        if not is_local:
            time.sleep(2)  # chờ 2s giữa các batch khi gọi cloud API

    # Lưu cache cho các lần chạy sau
    try:
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(cache_dir))
        print(f"💾 Đã lưu FAISS vectorstore vào cache: {cache_dir}")
    except Exception as e:
        print(f"⚠️ Không thể lưu cache: {e}")

    print("✅ FAISS vectorstore đã sẵn sàng.")
    return vectorstore
