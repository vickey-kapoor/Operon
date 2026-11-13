write_file(path, content): create a file or replace its whole text.
- Use it when the task asks for a new file, or to rewrite a file whose full new text you know.
- It replaces the whole file: read the file first and keep every line the task doesn't change.
- If the task asks you to use an app (for example "edit it with Notepad"), use the app.
- Example: write_file(path="Documents/summary.txt", content="Total: 42.00\n").
