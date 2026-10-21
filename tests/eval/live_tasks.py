"""Real web tasks for the live benchmark (``python -m tests.eval.live_bench``).

Each task names what the final screen must show (``check``). A judge model
reads the run's final screenshot against it, so a run's own "succeeded" claim
is never taken on trust. Sites are public, need no account, and are either
built for automation practice or stable reference pages; no task submits
anything to a real service.

To add a task: pick a stable page, write the intent the way a user would, and
make ``check`` something a person could confirm from one screenshot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

INTERNET = "https://the-internet.herokuapp.com"


@dataclass(frozen=True)
class LiveTask:
    name: str
    category: str
    intent: str
    start_url: str | None  # None: start from the current screen (desktop tasks)
    check: str
    max_steps: int = 15
    tags: frozenset[str] = field(default_factory=frozenset)


SMOKE = frozenset({"smoke"})

TASKS: list[LiveTask] = [
    # --- navigation ---------------------------------------------------------
    LiveTask(
        "click_link", "navigation",
        "Click the 'Learn more' link on this page",
        "https://example.com",
        "The IANA 'Example Domains' page is showing, not the example.com page.",
        tags=SMOKE,
    ),
    LiveTask(
        "go_back", "navigation",
        "Open the 'Talk' tab, then go back to the article",
        "https://en.wikipedia.org/wiki/Alan_Turing",
        "The Alan Turing article (not its Talk page) is showing, with the 'Article' tab selected.",
        tags=SMOKE,
    ),
    LiveTask(
        "toc_section", "navigation",
        "Use the table of contents to jump to the 'Commemoration' section",
        "https://en.wikipedia.org/wiki/Ada_Lovelace",
        "The 'Commemoration' section heading of the Ada Lovelace article is visible near the top of the view.",
    ),
    LiveTask(
        "hn_new", "navigation",
        "Open the 'new' stories page",
        "https://news.ycombinator.com/",
        "Hacker News is showing the 'new' page ('new' is highlighted in the top bar).",
    ),
    LiveTask(
        "hn_second_page", "navigation",
        "Go to the second page of stories",
        "https://news.ycombinator.com/",
        "Hacker News is showing the second page of front-page stories, numbered from 31.",
    ),
    LiveTask(
        "python_whatsnew", "navigation",
        "Open the 'What's new' page for this Python version",
        "https://docs.python.org/3/",
        "A Python 'What's New In Python 3.x' documentation page is showing.",
    ),
    LiveTask(
        "books_category", "navigation",
        "Open the Mystery category",
        "https://books.toscrape.com/",
        "The Mystery category page is showing, with 'Mystery' as the heading and 32 results.",
    ),
    LiveTask(
        "quotes_page_two", "navigation",
        "Go to the next page of quotes",
        "https://quotes.toscrape.com/",
        "The second page of quotes is showing, not the first. Page 2 has quotes by Bob Marley, Dr. Seuss, "
        "Douglas Adams, Elie Wiesel, Friedrich Nietzsche, Mark Twain and Allen Saunders, none of whom are on "
        "page 1; page 1 has Thomas A. Edison, Jane Austen, Steve Martin, André Gide and Eleanor Roosevelt.",
    ),
    # --- search -------------------------------------------------------------
    LiveTask(
        "search_enter", "search",
        "Search Wikipedia for Alan Turing and open his article",
        "https://en.wikipedia.org/wiki/Main_Page",
        "The Wikipedia article titled 'Alan Turing' is showing.",
        tags=SMOKE,
    ),
    LiveTask(
        "wiki_search_other", "search",
        "Search Wikipedia for Grace Hopper and open her article",
        "https://en.wikipedia.org/wiki/Main_Page",
        "The Wikipedia article titled 'Grace Hopper' is showing.",
    ),
    LiveTask(
        "pypi_search", "search",
        "Search PyPI for the httpx package and open its project page",
        "https://pypi.org/",
        "The PyPI project page for 'httpx' is showing, with its version and an install command.",
    ),
    LiveTask(
        "python_docs_search", "search",
        "Search the Python docs for asyncio.gather",
        "https://docs.python.org/3/",
        "Python docs search results for 'asyncio.gather' are showing, or the asyncio docs at asyncio.gather.",
    ),
    LiveTask(
        "mdn_search", "search",
        "Find the MDN reference page for Array.prototype.map",
        "https://developer.mozilla.org/en-US/",
        "The MDN reference page titled 'Array.prototype.map()' is showing.",
    ),
    LiveTask(
        "gutenberg_search", "search",
        "Find the book Pride and Prejudice and open its page",
        "https://www.gutenberg.org/",
        "The Project Gutenberg page for 'Pride and Prejudice' by Jane Austen is showing.",
    ),
    LiveTask(
        "duckduckgo_search", "search",
        "Search for 'vision language model' on this search engine",
        "https://duckduckgo.com/",
        "A DuckDuckGo results page for 'vision language model' is showing.",
    ),
    LiveTask(
        "osm_search", "search",
        "Search the map for the Eiffel Tower",
        "https://www.openstreetmap.org/",
        "OpenStreetMap shows search results for the Eiffel Tower, or the map centred on it in Paris.",
    ),
    # --- reading: find a fact and leave it on screen -------------------------
    LiveTask(
        "book_price", "reading",
        "Open the book 'A Light in the Attic' to see its price",
        "https://books.toscrape.com/",
        "The product page for 'A Light in the Attic' is showing its price of £51.77.",
    ),
    LiveTask(
        "quotes_by_tag", "reading",
        "Show only quotes tagged 'love'",
        "https://quotes.toscrape.com/",
        "Only quotes tagged 'love' are listed: the 'Viewing tag: love' header is visible, or every quote "
        "on screen carries the 'love' tag.",
    ),
    LiveTask(
        "wiki_infobox_fact", "reading",
        "Open the article on the Eiffel Tower so its height is visible",
        "https://en.wikipedia.org/wiki/Main_Page",
        "The Eiffel Tower Wikipedia article is showing, with its height visible in the infobox or text.",
    ),
    LiveTask(
        "scroll_down", "reading",
        "Scroll down to the References section of this article",
        "https://en.wikipedia.org/wiki/Alan_Turing",
        "The 'References' section heading of the Alan Turing article is visible.",
        tags=SMOKE,
    ),
    LiveTask(
        "scroll_to_footer", "reading",
        "Scroll to the bottom of this page",
        "https://docs.python.org/3/",
        "The footer of the Python docs page is visible (copyright or 'last updated' line).",
    ),
    # --- forms --------------------------------------------------------------
    LiveTask(
        "multi_field_form", "forms",
        "Fill in the customer name Ada Lovelace and the email ada@example.com, then submit the order",
        "https://httpbin.org/forms/post",
        "The form was submitted: the response page shows JSON containing 'Ada Lovelace' and 'ada@example.com'.",
        tags=SMOKE,
    ),
    LiveTask(
        "form_radio_checkbox", "forms",
        "Choose a Large pizza with Mushroom and Onion toppings, name it for Grace, and submit the order",
        "https://httpbin.org/forms/post",
        "The submitted form's JSON shows size 'large', toppings including 'mushroom' and 'onion', and custname 'Grace'.",
    ),
    LiveTask(
        "login_internet", "forms",
        "Log in with username tomsmith and password SuperSecretPassword!",
        f"{INTERNET}/login",
        "The Secure Area page is showing with the message 'You logged into a secure area!'.",
    ),
    LiveTask(
        "login_practice", "forms",
        "Log in with username student and password Password123",
        "https://practicetestautomation.com/practice-test-login/",
        "The 'Logged In Successfully' page is showing.",
    ),
    LiveTask(
        "login_quotes", "forms",
        "Log in with any username and password",
        "https://quotes.toscrape.com/login",
        "The quotes page is showing with a 'Logout' link instead of 'Login'.",
    ),
    LiveTask(
        "dropdown", "forms",
        "Select Option 2 in the dropdown",
        f"{INTERNET}/dropdown",
        "The dropdown shows 'Option 2' as the selected value.",
    ),
    LiveTask(
        "checkboxes", "forms",
        "Make sure both checkboxes are checked",
        f"{INTERNET}/checkboxes",
        "Both checkboxes are checked.",
    ),
    LiveTask(
        "number_input", "forms",
        "Type the number 42 into the input",
        f"{INTERNET}/inputs",
        "The number input contains 42.",
    ),
    LiveTask(
        "key_press", "forms",
        "Press the Tab key in the input field",
        f"{INTERNET}/key_presses",
        "The page reports 'You entered: TAB'.",
    ),
    # --- multi-step flows ---------------------------------------------------
    LiveTask(
        "todo_add_complete", "flows",
        "Add two todos, 'buy milk' and 'walk dog', then mark 'buy milk' as completed",
        "https://demo.playwright.dev/todomvc/",
        "The todo list has 'buy milk' (completed, struck through) and 'walk dog' (not completed).",
    ),
    LiveTask(
        "todo_filter", "flows",
        "Add the todos 'one' and 'two', complete 'one', then show only active todos",
        "https://demo.playwright.dev/todomvc/",
        "The 'Active' filter is selected and only 'two' is listed.",
        max_steps=20,
    ),
    LiveTask(
        "sauce_add_to_cart", "flows",
        "Log in with standard_user / secret_sauce and add the Sauce Labs Backpack to the cart",
        "https://www.saucedemo.com/",
        "The products page shows the cart badge at 1 (the Backpack's button reads 'Remove'), or the cart lists the Backpack.",
    ),
    LiveTask(
        "sauce_checkout", "flows",
        "Log in with standard_user / secret_sauce, buy the Sauce Labs Bike Light, and complete checkout as Ada Lovelace, postal code 12345",
        "https://www.saucedemo.com/",
        "The checkout complete page shows 'Thank you for your order!'.",
        max_steps=25,
    ),
    LiveTask(
        "add_remove", "flows",
        "Add three elements, then delete one of them",
        f"{INTERNET}/add_remove_elements/",
        "Exactly two 'Delete' buttons are showing.",
    ),
    LiveTask(
        "shop_laptops", "flows",
        "Open the Laptops category under Computers",
        "https://webscraper.io/test-sites/e-commerce/allinone",
        "The webscraper.io test site shows the Laptops listing (heading 'Computers / Laptops').",
    ),
    # --- dynamic pages and interaction kinds --------------------------------
    LiveTask(
        "dynamic_loading", "dynamic",
        "Click Start and wait for the result to appear",
        f"{INTERNET}/dynamic_loading/2",
        "'Hello World!' is showing on the page.",
    ),
    LiveTask(
        "dynamic_controls", "dynamic",
        "Remove the checkbox using the button on the page",
        f"{INTERNET}/dynamic_controls",
        "The message 'It's gone!' is showing and the checkbox is no longer there.",
    ),
    LiveTask(
        "enable_input", "dynamic",
        "Enable the text field and type hello into it",
        f"{INTERNET}/dynamic_controls",
        "The text field is enabled ('It's enabled!') and contains 'hello'.",
    ),
    LiveTask(
        "hover_reveal", "dynamic",
        "Hover over the first profile picture to reveal its name",
        f"{INTERNET}/hovers",
        "The caption 'name: user1' is visible under the first picture.",
    ),
    LiveTask(
        "close_modal", "dynamic",
        "Close the advertisement popup",
        f"{INTERNET}/entry_ad",
        "The 'Entry Ad' page is showing with no modal window open.",
    ),
    LiveTask(
        "slider", "dynamic",
        "Set the slider to 4",
        f"{INTERNET}/horizontal_slider",
        "The slider's value reads 4.",
    ),
    LiveTask(
        "drag_drop", "dynamic",
        "Drag box A onto box B",
        f"{INTERNET}/drag_and_drop",
        "The boxes have swapped: the left box reads 'B' and the right box reads 'A'.",
    ),
    LiveTask(
        "sort_table", "dynamic",
        "Sort the first table by last name",
        f"{INTERNET}/tables",
        "Example 1's rows are sorted by Last Name (Bach, Conway, Doe, Smith from the top, or the reverse).",
    ),
]
