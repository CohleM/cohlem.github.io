import os
import re
import shutil

# Root directory containing _posts
POSTS_DIR = "_posts"

# Regex to match Markdown image syntax: ![alt](filename.ext)
IMAGE_PATTERN = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')

def update_markdown_images(md_path, folder_name):
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Backup original file
    backup_path = md_path + ".bak"
    shutil.copy(md_path, backup_path)

    def replacer(match):
        alt_text = match.group(1).strip()
        filename = os.path.basename(match.group(2).strip())
        # Only rewrite relative images (no / or http)
        if "/" in filename or filename.startswith("http"):
            return match.group(0)
        new_path = f"/assets/images/{folder_name}/{filename}"
        return f"![{alt_text}]({new_path})"

    new_content = IMAGE_PATTERN.sub(replacer, content)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"✅ Updated: {md_path}")
    print(f"   Backup saved as: {backup_path}")

def main():
    for root, dirs, files in os.walk(POSTS_DIR):
        # Skip the root _posts folder itself
        if root == POSTS_DIR:
            continue

        folder_name = os.path.basename(root)

        for file in files:
            if file.endswith(".md"):
                md_path = os.path.join(root, file)
                update_markdown_images(md_path, folder_name)

if __name__ == "__main__":
    main()
