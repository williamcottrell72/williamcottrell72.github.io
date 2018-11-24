---
layout: post
title: Scraping Amazon
description: Battle With the Behemoth
---

## Scraping Amazon

For those who want data (who doesn't?) and who can't find what they're looking for on Kaggle, there is often the possibility of scraping the data from the web.  'Scraping' may sound intimidating at first, but it is really just the process of automating the web-search that you might normally do to look something up online.  Fortunately, there are many useful tools in Python to make this easier.  In todays post, I'll discuss how to scrape Amazon from start to finish.  I'll also go over an analysis of the data.  This is just going to be a broad overview; more details will be provided in subsequent posts.


### Motivation

Let's say I'm selling widgets on Amazon.  Well, actually, I WAS selling widgets on Amazon, so this is not too hard to imagine.  I want to know how to best design my seller's page.  I.e., how do I optimize the parameters of the page layout in order to maximize profit?

Of course, I could go into one of many Amazon seller chat forums where this precise question is often discussed.  There are of course standard recommendations.  However, I don't trust humans, I only trust machines.  WWMD? (What would machines do?)

Now, I could consider using the Amazon API to get the data I want.  An API is like filling out a FOIA for a website.  And, like an FOIA, you won't necessarily be able to get the data you want, when you want it.  You are basically constrained by what they are willing to give you and you have to play by their rules.  In the case of Amazon, one has to sign up as a seller/developer, get a KEY and, most annoyingly, promise to post Amazons junk on your blog.  I would never want to corrupt the purity of this moderately-OK blog, so that was a no-go.

So, we are down to scraping.  The data I want is right on the seller page so I know this should be possible.  In general, if you can see it, you can scrape it.  Of course, the API might actually let you get information that is invisible to the normal user, but we don't need that anyway.

In order to scrape Amazon I used a combination of Selenium and BeautifulSoup.  These are both Python modules, so, my first two lines of code will look like:

```python
from bs4 import BeautifulSoup
from selenium import webdriver
from selnium.webdriver.common.keys import Keys
```

Selenium is a module that let's python control web-browser just like a human would.  In other words, you are telling the browser step-by-step what buttons to push and what to type in the web-site.

BeautifulSoup, on the other hand, just takes the raw HTML and parses it in order to find useful information.  It also helps format the HTML so that it is human readable.

For websites written in HTML BeautifulSoup alone would have sufficed.  However, many sights are written as JavaScript which is only rendered as HTML when opened.  So, basically, the strategy is to use selenium to open a page and the BeautifulSoup to read it.

Let's review the basic features of each:

### Selenium

Again, selenium just allows you to control the web-browser with python.  To use it, you must tell Python exactly where the driver for your browser lives.  First, you need to find where that was installed for your system.  For me it is in 'Downloads', for you it might be in Applications.  If you are using Chrome then the file you want will be called 'chromedriver'.  Once you find the path, you'll need something like:

```python
chromedriver = "path/chromedriver"
os.environ["webdriver.chrome.driver"] = chromedriver
```

Now, if you've gotten this far and you have a url that you want to read then the rest is easy.  Just fead the url to selenium and convert it to soup!

```python
driver.get(url)
soup=BeautifulSoup(driver.page_source,'html.parser')
```
Now, 'soup' is an HTML like object which can be parsed with bs4.  One useful method that can be called on 'soup' is 'find_all' which returns all objects with specific properties.  For instance,

```python
soup.find_all('span', class='...')
```

will find all the objects with header 'span' and of class '...'.  To figure out what parts of the page you're actually looking for it is convenient to go and open the browser and use the 'DevelopmentTools' feature.  This will allow you to display the HTML of whatever you click or hover over.  You then need to put those specs into the 'find_all' method.  There are more tools for parsing, but that is all that we will need for the moment.

### Beautiful Soup

(brief description of bs)

## Problems

Now we are ready to sccrape!

Of course, in datascience things are never that easy.  Amazon doesn't want us scraping them.  To see what, precisely, they do not like one can consult the .robots file.  In any case, we need to adopt some defensive measures.

1) Rotate IPs with _expressvpn_
2) Rotate user agent.
3) Insert randomized sleeps.
4) Randomize order of search, etc.

...need more time to finish...

...next blog will be about the data.
