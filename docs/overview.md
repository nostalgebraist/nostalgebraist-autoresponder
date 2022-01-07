## Technical overview

This doc contains a overview of what this codebase does.  It is up to date as of early January 2022.

### Basics

This repo is the codebase for my tumblr bot "nostalgebraist-autoresponder," commonly known as "Frank."

Using the tumblr API, the code operates a tumblr blog, also called [nostalgebraist-autoresponder](https://nostalgebraist-autoresponder.tumblr.com/).  This blog is a sideblog of my tumblr account; my main blog is [nostalgebraist](https://nostalgebraist.tumblr.com/).

The core of the bot is a [python script](https://github.com/nostalgebraist/nostalgebraist-autoresponder/blob/docs-reference-commit/src/tumbl.py) which runs continuously.  This script runs an infinite loop, where each iteration of the loop executes a specific sequence of steps, e.g. "check asks and respond to them" and "check the dash."  I refer to this as the **main loop**.

The rest of the codebase exists to implement a wide variety of functionality which the main loop uses as needed to parse, interpret, and create tumblr content.

### The internal compute API

The code is designed to spread tasks across several machines.  Specifically, running the bot involves:

- A **main machine**.  This machine is responsible for running the main loop, making tumblr API requests, various lightweight tasks, and data persistence.
- One ore more **ML machines**.  These are responsible for running tasks that need a GPU to achieve reasonable performance.  Here, that means running machine learning models.

The main machine has a small compute footprint, and can run on a very cheap cloud computer.  (For a long time, it ran on my laptop.)  This means we can rely on it to have a static IP.

By contrast, the ML machines require high-performance GPUs, which are a precious resource.  Obtaining this resource persistently in the cloud is very expensive.  It is much cheaper to obtain it ephemerally (e.g. spot instances, Colab hosted runtimes).  Thus, the code treats "ML machine" as a role rather than an identity: the specific computers playing this role may vary over time, as will their IPs.

How can we make sure the main machine and the ML machines know how to reach each other?  In a conventional system of this form, the emphemeral IPs of the ML machines might be bound to be persistent hostnames.  For better or for worse, this codebase does _not_ go this route.  Instead, it uses an eccentric API design that avoids ever sending requests _to_ the ML machines.

#### Life-cycle of an ML task

Here's how it works:

The main machine runs two python processes: the main loop itself, and a [**bridge service**](https://github.com/nostalgebraist/nostalgebraist-autoresponder/blob/docs-reference-commit/src/api_ml/bridge_service.py).  The latter is a stateful web service which the ML machines poll at some regular interval with `/pollml` GET requests.

When the main loop process needs to use an ML model, it makes a `/requestml` POST request to the bridge service.  This pushes the desired ML task onto a queue stored in the bridge service.  The main loop process now begins to poll the bridge service with `/getresult` requests.

The next time an ML machine polls the bridge service with GET `/pollml`, it receives the queue and executes the tasks on it.  When it's done, it sends the results in a a POST `/pollml` request to the bridge service.  The bridge service records the result, and pushes the task off the queue.

On the next `/getresult` request from the main loop, the bridge service informs the main loop that the task is complete and sends over the results.

The description above is accurate in broad strokes, but note that the actual code is somewhat more complicated.  For example:

- A common ML task is "write `N` different versions of the same post."  In this case, the ML machines do not wait until they have completed the entire task before sending results back.
  - Instead, they send each text they write immediately upon its completion, and the bridge service "keeps the task alive" until it has received `N` or more texts.
  - This parallelizes the task across all ML machines currently available.
- There are two kinds of ML machine: the ones that deal with text, and the ones that do image generation tasks.  Image generation tasks have their own version of each bridge endpoint.

#### Layers and connectors

Code that runs on the ML machines generally lives in the `src/ml/` directory, and the core code lives in files prefixed with `ml_layer_`.

The full scripts that runs on ML machines are private and not included in this repo, but they essentially consist of installing dependencies, importing everything from an `ml_layer_` module, and running the module's `loop_poll` function.

When the main loop executes ML tasks, it uses an interface that abstracts away the details of the bridge service.  This is implemented in files suffixed with `_connector.py`.

For instance, `ml_connector.py` [provides classes](https://github.com/nostalgebraist/nostalgebraist-autoresponder/blob/docs-reference-commit/src/api_ml/ml_connector.py#L82-L129) that look to the caller like local instances of the actual ML models, but which operate "under the hood" by interacting with the bridge service.

(Unforunately, `ml_connector.py` also contains a large amount of main loop business logic.  In a future refactor, I would want to separate these concerns.)

### The main loop

A single iteration of the main loop [consists](https://github.com/nostalgebraist/nostalgebraist-autoresponder/blob/docs-reference-commit/src/tumbl.py#L2714-L2820) of the following steps in order:

1. Check drafts, respond to content moderation tags if needed
2. Check asks, write and post responses if needed
3. Check for new reblogs, replies and mentions, write and post responses if needed
4. Repeat step 1 (content moderation check)
5. Repeat step 2 (asks check)
6. Read new posts on the dash, decide whether to reblog them, write and post responses if needed
7. Repeat step 2 (asks check)
8. Repeat step 1 (content moderation check)
9. Check queue, write and queue new original posts if there are too few in the queue
10. Repeat step 1 (content moderation check)
11. Wait for 60 seconds, or longer in some cases

_(Note that step 3 is implemented by looping over the bot's last 250 posts, checking the notes of each one, and looking for user input that has not been marked "handled" in the bot's internal state.  This is expensive in terms of API calls, but was the only way to do it when I started this project._

_More recently, tumblr has introduced a notifications endpoint, much like a human user's activity page, which is a much nicer way to do it.  I do use this endpoint to check for mentions, which was previously impossible, but I still use the old approach for reblog and reply checking.)_

Steps that can modify the bot's internal state will often save state data to disk after running; for example, this happens after every asks check if any new asks were handled.

Some steps require the bot to read content from tumblr (asks, reblogs/replies, dash), or create new content to post (asks, reblogs/replies, dash, and queue).

When it "reads" tumblr posts, the main loop first fetches them from the tumblr API.  In most cases, it then uses data in the API response to decide what to do next.  These decisions include things like:

- To spread out user demand, the steps that check asks/reblogs/replies will only respond to one such message per tumblr user.   If there's more than one new ask/reblog/reply from a user, the oldest one is handled first.
- To spread out user demand, at most 5 asks can be handled per asks check.  If there are more than five (after applying the previous rule), the oldest ones are handled first.
- The bot can only reblog a post from the dash if it meets a list of conditions.  For example, the bot never reblogs posts which the user on the dash has reblogged without their own added commentary.

After applying these rules, the main loop may decide that it needs to write new post(s).

### Writing posts

This is the bot's most fundamental job, and is quite complex.

Schematically, writing a post looks like:

1. Preparing to write
    - If we are replying to something (i.e. not writing an original post for the queue), convert the tumblr API response to a standardized text representation, using the **text munging layer**.
    - If we are replying to something, run ML tasks to produce **sentiment** and **autoreview** scores for the input.  (Respectively, "how happy/sad is this input?" and "how likely is this input to be 'problematic' or otherwise 'unpostable'?")
    -  Determine the current **mood**.
    -  Estimate how many candidate posts will be rejected as incompatible with current the mood, and choose the number of target candidates `N` to account for this.
2. Writing candidates
    - Run an ML task to write `N` candidate posts.
3. Candidate assessment
    - Run ML tasks to produce **selector** ("how much will people like this?"), sentiment, and autoreview scores for each candidate.
    - Remove candidates inconsistent with the current mood.
    - Pick one post from the remaining candidates based on selector scores, typically with an eps-greedy strategy.
4. Automated content moderation
    - Decide whether to post it automatically, send it to content moderation, or flat-out reject it.  The rules behind this decision use the autoreview score together with a manually curated word list.
5.  Making the post
     - Parse the the chosen post into an HTML main body and a list of tags.
     - If the main body contains text representations of images, generate images for each one, upload them to tumblr, and replace the text representations with corresponding `<img>` tags.
     - If we are OK to post automatically (step 4), send a tumblr API request to publish or queue the post.  If it's being sent to content moderation, save it to drafts instead.  If we're flat-out rejecting it, save it to drafts with a special rejection tag.

Most of the nontrivial parts of the codebase are used somewhere in this procedure.  Below, I'll describe some of these.

### The text munging layer

The tumblr API returns structured data.  However, the ML model that "reads" and "writes" post is an autoregressive language model and operates only on _text_.  To connect the two, I define a standard textual representation for a tumblr post.

Within a post, this representation is simply HTML with a stripped-down set of tags.  The boundaries between posts are denoted by special delimeters containing info about usernames, position and role in the conversation, and (for the final post) tags and posting date.

An ask and response in this format might look like this:

```
<|endoftext|> Conversation between example-username and nostalgebraist-autoresponder | 2 posts |
#1 example-username asked:

 How's it going?

#2 nostalgebraist-autoresponder responded:

 Written 11 PM January 2021 | nostalgebraist-autoresponder's tags: #example tag, #other example tag
 
 Pretty good!  <i>Great</i>, actually!<|endoftext|>
```

To write the bot's response, the main loop would first convert the content of the ask into a prompt for the language model:

```
<|endoftext|> Conversation between example-username and nostalgebraist-autoresponder | 2 posts |
#1 example-username asked:

 How's it going?

#2 nostalgebraist-autoresponder responded:

 Written 11 PM January 2021 | nostalgebraist-autoresponder's tags:
```

The language model then writes the rest of the text.  In this example, that's:

```
 #example tag, #other example tag
 
 Pretty good!  <i>Great</i>, actually!<|endoftext|>
```

#### Converting posts to text

Tumblr has two post formats, legacy and NPF.

Legacy expresses both structure and content as HTML: for example, the boundaries between posts in a conversation are conveyed through nested blockquotes.  A parser for this format must extract this implicit structure, distinguishing these "structural blockquotes" from the "content blockquotes" that can be used to style text within a post.  (Legacy API responses include a sort-of-redundnant field called the "trail" which conveys tumblr's opinion about structure vs. content, but it is undocumented and unreliable.)

NPF is a structured JSON format similar to the internal formats of DraftJS or WordPress.  It distinguishes posts from one another explicitly.  Within a post, it uses a domain-specific language of "content blocks" instead of HTML.

Posts can be created in either format, and requested in either format.  If you request a legacy post in NPF, or vice versa, tumblr attempts to translate between formats.  This translation is usually reliable for legacy --> NPF, but quite flaky for NPF --> legacy.

I originally produced the text format by parsing HTML from legacy-format tumblr API responses.  This had two large downsides:

1. Many posts are created as NPF, and NPF's adoption grew over this bot's lifetime.  Legacy API representations of NPF posts are very unreliable.
2. I had two textual formats (HTML from the API, and the internal text format), but no structured format, even internally.  This made transformations on posts (e.g. simulating what a post would look like if the bot's response was included) into text parsing problems, which required increasingly complex regex work and was eventually unsustainable.

These days, I request posts in NPF, and the code uses its data model of NPF as its fundamental data model for posts.  That is, transformations on posts [happen on NPF objects](https://github.com/nostalgebraist/nostalgebraist-autoresponder/blob/docs-reference-commit/src/tumblr_to_text/nwo_munging.py), not on text.

To produce the text representation from NPF, I use my own [NPF to legacy conversion code](https://github.com/nostalgebraist/nostalgebraist-autoresponder/blob/docs-reference-commit/src/api_tumblr/tumblr_parsing.py#L701-L710) to translate content within posts into a legacy-like HTML format.  I then construct the delimeters between posts by [translating the relevant parts of the NPF structure](https://github.com/nostalgebraist/nostalgebraist-autoresponder/blob/docs-reference-commit/src/tumblr_to_text/nwo.py#L40-L87).

#### Reading images

When converting a tumblr post that contains images, the main process fetches the images and sends the to the DetectText endpoint of AWS's Rekognition service.  The raw response is distilled into a text representation, which is substituted into the text representation of the post.  Most of the relevant code is [here](https://github.com/nostalgebraist/nostalgebraist-autoresponder/blob/docs-reference-commit/src/multimodal/image_analysis.py).

These calls are cached, both by image URL and (on a URL cache miss) by a hash of the image bytes.  The latter mechanism accounts for the fact that some images are uploaded to tumblr many times under different URLs.

### The mood

The bot's "mood" is a scalar variable that evolves continuously over time.  It is the sum of two parts:

- a base value, constant within a calendar day, randomly reset at midnight to one of a few possible values
- a user-impact signal, whose evolution is governed by a Linear Time-Invariant system of differential equations

When the bot responds to user input, the main process calculates a mood effect.  This is a weighted average of 

- the sentiment score of the input (did the user's words sound positive or negative?), and
- a summary statistic of the sentiment scores for all the candidate responses (did the user say something that would receive a positive-sounding response, or a negative-sounding one?)

The resulting value affects the user-impact signal as a Dirac delta kick to its first derivative.  That is, when the effect happens, the mood's rate of increase or decrease immediately changes by a proportional amount.  In addition to these effects, the rate naturally relaxes exponentially towards zero with a time constant of several hours.

The mood is re-calculated at most once every 10 minutes.  When it is required for a decision, if there's a fresh value from the last ten minutes, this "slightly stale" value is used.  If the saved value is over 10 minutes old, a new value is computed by numerically integrating the equations.  This calculation only uses user effects from the last 2.5 days, as the exponential decay means older effects have negligble impact.

The code for this is mostly [here](https://github.com/nostalgebraist/nostalgebraist-autoresponder/blob/docs-reference-commit/src/feels/mood_dynamic.py).

#### Mood consistency

The mood is used when making posts.  The mood value determines an acceptable range of sentiment scores.  I define a mapping from mood values to ranges for [a short list of "basic moods"](https://github.com/nostalgebraist/nostalgebraist-autoresponder/blob/docs-reference-commit/src/feels/mood.py#L61-L143), and interpolate the ranges linearly when the current mood lies between two of these.

During post writing, a large number of candidate posts are generated.  Any of these that fall outside the range are discarded, though their scores are still included when calculating mood effects (see above).

#### Content moderation

The language model is capable of writing some very bad things, e.g. racist material.

The code tries to identify this kind of content -- or content that _might_ be this kind -- before posting it, and will save the post as a draft if it's "worried" about it.  I manually check the bot's drafts folder many times a day, and decide whether to post or delete the drafts.

Originally, this was done with a simple hand-curated word filter.  This gradually became unsustainable, with lots of posts ending up in drafts for silly reasons, while others slipped through because they used words or phrases I hadn't thought of in advance.

To improve the situation, I trained a machine learning model on the logs from my own decisions about past drafts.  This model renders a judgment on each post the bot wants to make.

The model's judgment is used together with the original word list: posts that don't trip the word filter are automatically posted unless the model is pretty sure they're "bad," while posts that do trip the word filter are sent to drafts unless the model is pretty sure they're "OK."  The code for this logic is [here](https://github.com/nostalgebraist/nostalgebraist-autoresponder/blob/docs-reference-commit/src/tumbl.py#L376-L553).

When I want to reject a post in drafts, I generally still want the bot to write _some_ response to the user's input, just not this one.  This means I must communicate to the main loop that it should no longer view the user input as "already answered," and should answer it again from scratch.

To do this, I add a special tag to the draft post, which the bot is not allowed to use in published posts.  During the "content moderation check" parts of the main loop, posts with this tag are identified and "reset."  This looks something like

- Asks: change the state of the post from "draft" to "submission," which is the state of asks in a user's inbox.  During the next asks check, the main loop will see the ask and respond again.
- Reblogs, replies, mentions: delete the draft and un-set the "already handled" flag for the input in the bot's state.  During the next reblog/reply/mentions check, the main loop will respond again

Original posts and dash reblogs can simply be deleted if they're unacceptable.

When this system causes a post to be written multiple times, only the first write has mood effects.  (Before I enforced this, the bot could become very unhappy as it tried to respond over and over to something borderline-unpublishable.)

### Odds and ends

#### Tumblr API load management

Early on, I had a lot of trouble staying under the tumblr API rate limits.  Improvements to the call logic over time have rendered this mostly a non-issue, but the code still uses some features that I developed during this period.

There are two rate limits, daily and hourly.  The daily one, which resets every 24 hours, was the source of my early issues.

Given a fixed budget of API calls per 24 hours, the code tries to spread the calls out evenly, rather than spending them all quickly and then having to halt until the next reset.  It does this with a control system.  The code

1. records the number of calls used in each iteration of the main loop
2. calculates how many times the main loop could be run in a 24-hour interval, if every iteration used the median number of calls from the last 30 loops of data
3. calculates an upper bound on how many times we _want_ to run the main loop in the next 24 hours, by assuming the main loop is instantaneous and only counting the "sleep time" between iterations
4. if the number we can support (2) is less than the number we want to do (3), randomly skips the main loop some fraction of the time

This mechanism rarely kicks in these days, but does in the occasional edge case.

Additionally, some individual API requests are cached -- although I've removed much of this over time, as the calls have become both less repetitive and more efficient.

#### Original posts: mood projection

Original posts are queued well in advance, using tumblr's queue mechanism.  This makes it impossible to ensure they are consistent with the bot's mood at the time they are posted, since it will be affected by future input from users.

I do screen these posts for mood like other posts, though.  To get a best guess for the future mood value, the code evolves the mood system forward to the intended posting time, pretending that no user input will occur between now and then.

#### Original posts: retention

Writing an original post is much like writing a response: we generate many candidates, reject some based on mood, and select one of the remaining ones.

However, doing this naively means discarding a lot of good posts that could be used later.  Original posts are fungible in a way that responses are not, since there's no pressure to post them at any particular time.  A post that gets rejected for mood consistency might be just right given the mood at some other, later time.  And if we get several great posts consistent with the current mood, we still have to pick only one of them.

To make this less wasteful, the bot's state includes a "retention set" of original posts that have not been made.  Each time the bot writes an original post, it uses this as a starting set of candidates, to which the newly-written candidates are added.  After selecting the next original post, any candidate with very high selector scores that were _not_ chosen are added to the set.

#### Private config

Some data needed to run the bot is stuff I don't want to share publically, such as API keys and the "bad words list" for content moderation.  This data is stored in a json file in a private GCS bucket, rather than in this repo.  The contents of this file are loaded and modeled by the code [here](https://github.com/nostalgebraist/nostalgebraist-autoresponder/blob/docs-reference-commit/src/config/bot_config.py).

#### Deciding which posts to reblog from dash


### The ML models

### Data persistence
