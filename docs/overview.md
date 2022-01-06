## Technical overview

This doc contains a high-level description of how the bot is implemented.  It is up to date as of early January 2022.

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

By contrast, the ML machines require high-performance GPUs, which are a precious resources.  Obtaining this resource persistently in the cloud is very expensive.  It is much cheaper to obtain it ephemerally (e.g. spot instances, Colab hosted runtimes).  Thus, the code treats "ML machine" as a role rather than an identity: the specific computers playing this role may vary over time, as will their IPs.

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

### The main loop

### The text munging layer

### The ML models

### Data persistence
