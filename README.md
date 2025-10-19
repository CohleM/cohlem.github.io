This profile was self-designed with the help of claude using jekyll.

If you want to use this as a template then follow these steps:

1. First clone this repo

2. Install the requirements, use this [reference](https://jekyllrb.com/docs/installation/)

3. ```bash
   bundle install
   ```

4. Only change files in these two folders `_data` and `_posts`

   **_\_data_**
   Update `homepage.yml` with your own details (i.e projects/ publications)

   **\_posts**
   Use this for your blogs. Each folder will be a blog. Folders inside it need to be in a specific format i.e YYYY-MM-DD-blog-title and inside that folder your markdown file should also have the same name but with `.md` extension i.e YYYY-MM-DD-blog-title.md. You can add all the images for this blog within the folder YYYY-MM-DD-blog-title

That's all you need to know if you simply want to use it without any design/page changes.

If you want to change font color/sizes. Change the CSS file in `assets/css/main.scss`

If you want to add a new page then create a .html file at the root folder and add the link to that page in the nav element in `_layouts/default.html`

If you want more changes then use claude code/codex with this website as a context.
