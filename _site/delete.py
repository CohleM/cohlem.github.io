# import os

# # Root directory containing your post folders
# POSTS_DIR = "_posts"

# # File extensions to delete (add more if needed)
# IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}

# def cleanup_post_folders():
#     for root, dirs, files in os.walk(POSTS_DIR):
#         # Skip the top-level _posts folder itself
#         if root == POSTS_DIR:
#             continue

#         for file in files:
#             file_path = os.path.join(root, file)
#             ext = os.path.splitext(file)[1].lower()

#             # Keep only .md files, delete image or any other files
#             if ext in IMAGE_EXTENSIONS or (ext and ext != ".md"):
#                 try:
#                     os.remove(file_path)
#                     print(f"🗑️  Deleted: {file_path}")
#                 except Exception as e:
#                     print(f"⚠️  Could not delete {file_path}: {e}")

#     print("\n✅ Cleanup complete — only .md files remain inside _posts/ folders.")

# if __name__ == "__main__":
#     cleanup_post_folders()


import os

# Root directory containing image folders
ASSETS_DIR = "assets/images"

def cleanup_markdown_files():
    deleted_count = 0

    for root, dirs, files in os.walk(ASSETS_DIR):
        for file in files:
            if file.lower().endswith(".md"):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"🗑️  Deleted: {file_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"⚠️  Could not delete {file_path}: {e}")

    if deleted_count == 0:
        print("✅ No .md files found — all clean!")
    else:
        print(f"\n✅ Cleanup complete — deleted {deleted_count} Markdown file(s).")

if __name__ == "__main__":
    cleanup_markdown_files()
