http_get(url): fetch a URL and get the status and body text back.
- Use it to read data from an API or a plain-text page. Use the browser for pages the user should see, and for anything that needs a login or a form.
- Redirects are not followed; the reply shows where one points.
- Example: http_get(url="https://api.example.com/orders/123").
