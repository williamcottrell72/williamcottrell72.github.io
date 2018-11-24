---
layout: post
title: Project Turnstile
description: Sounds Boring, but Actually Interesting!
---
Todays blog is about my first project for the Metis datascience bootcamp (SF 2018).
The premise of this project is that a group 'WomenTechWomenYes' (WTWY) is trying
to distribute flyers in support of their upcoming summer gala.  The group would like to find a good plan for when and where to distribute.  'We' are being hired to help them develop this strategy.

The group is based in New York and so we have lots of subway turnstile data to work with.  This will form the basis for our analysis.  The basic idea is that the more people leaving the subway at a given time the more people will accept fliers.  We are particularly interested in exits.  No-one can stop a New Yorker trying to catch a subway.

We are also interested in other factors which might affect when someone will accept a flier.  Since the target group is women in tech, we will want to focus on regions with more women.  Tech workers like coffee, so it might also help to look at stations near many coffee shops.  We could also look at tech companies, or any number of other factors.

What we really want is an easily replicable pipeline that will allow us to look for the locations of *any* type of establishment and then compute the distances to nearby subway stations.  Fortunately, the google places api will help us do this.  Basically, we will be able to enter in a plane English keyword, (e.g., 'Coffee Shops' or 'Apple Stores') and the goolgeplaces api will just spit out a list of addresses.  We can then convert these into a form that geopandas can recognize and make some nice plots.
Our goal is to do this for a few different keywords and then form an aggregate score based on the distances between subway stations and the output of our keyword searches.

The code for this project may be found here: [repo](https://github.com/williamcottrell72/project_benson).  My team consisted of myself, Xu, Alan, Auste and Chelan.  Much of the code in the repo is theirs, though I plan to clean it up and synthesize it better when I have more time.

#Implementation

##MTA Data

The MTA data is freely available from the MTA website.  The data is organized by week, with the startdate of the week being used in the page url.  This structure allows one to scrape multiple weeks using code of the form:

```python
def scrape(week_nums):
    url = "http://web.mta.info/developers/data/nyct/turnstile/turnstile_{}.txt"
    dfs = []
    for week_num in week_nums:
        file_url = url.format(week_num)
        dfs.append(pd.read_csv(file_url))
    return pd.concat(dfs)
```

Here, 'week_nums' is a list of properly formatted week numbers we've generated using the python datetime module.

Since there is quiet a bit of data on this site we should be a bit selective about how we scrape.  We are only interested in one particular month, the month before the gala, and so we could only scrape that month in some number of years.  In fact, this is what I tried at first.  Unfortunately, that led to an annoying problem. It turns out that the data only contains information about the cumulative number of exits up to a given time.  The easiest way to get the flux is to apply a .diff operatoin on a pandas dataframe.  However, if we have rows separated by a whole year the .diff will give an anomolously large value, which we will then have to figure out what to do with.

An easier way to deal with this is to scrape 3 months for each year, the month you care about and the months preceding and following.  Then, apply diff and drop the two extra months.  This will leave us with better data and we won't have to scrape everything.  (Of course, we really just need two extra weeks, not two extra months.)

After scraping the data we want to process it into a useful form.  We want to optimate 'flier acceptance likelihood' and there are two independent variables 'place' (MTA station) and 'time'.  Obviously, we don't want to just optimize both 'place' and 'time' simultaneously since then we will only have one place to go at one time per week.  So, what do we want to do? Maybe optimize  'time' when averaging over 'place' or 'place' averaging over 'time'?  Or, should we optimize one and then the other, or visa versa???

Of course, it is good to do all these things, but, we should keep in mind that time is really the most valuable commodity here.  The 'time' is really fixed by the volunteers schedules.  There are perhaps many potential volunteers who can only help at particular times.  Perhaps there are others that are free.  However, we get a huge benefit from bringing in volunteers whom otherwise would not be available.  And, when they are avaiable, we should be able to determine the best station.  Thus, what we really want is a function that reports back the best subway station as a function of time.

This is implemented in the 'Clean_and_Process.ipynb' file in the function 'activity_by_time'.  For the purpose of making maps, however, we just want to take some stations that 'on average' are the best.  This list is constructed in 'clean_df_cum'.  Specifically, we focus on the afternoon to avoid trapping too many people on their way to work.

The final results are heavily centered around Broadway, Times Square, etc.  A natural worry is that these are tourist and not likely to join the gala.  If there were more time this issue could be investigated further.

##Maps

As mentioned, the googleplaces api allows us to easily search for places based on a keyword.  To use this, we need to register and get a key:

```python
API_KEYAPI_KEY = input()
google_places = GooglePlaces(API_KEY)
gmaps = googlemaps.Client(key = API_KEY)
```

These functions are very user friendly since they allows us to type in normal English.  If you want the list of American restaurants with 3200m of Shenzhen, China for instance

```python
query_result=google_places.nearby_search(location='Shenzhen, China',keyword = 'American Restaurant', radius = 3200)
```

We want a list of Starbucks instead.

We can also use this method on the most popular MTA stations determined above.  Googleplaces allows us to get geojson codes for the stations, which can be fed into geopandas.  Overlaying this on a map showing gender percentages we get:

<img src="https://drive.google.com/uc?id=14TJGSCWEjuv5xRmt7oAkgR_IrdiWZY3i">

##Final Scores

Combining the various datasets we get a final score.  The choice of function used to combine these datasets if fairly arbitrary.  Ideally, more time would have been spent in deciding how to combine the various factors.  Oh well.  I have no more time to continue.  I hope you liked reading this.  Signing off.
