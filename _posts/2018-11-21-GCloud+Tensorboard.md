---
layout: post
title: Google CLoud Computing with Tensorboard
description: How to have your cake and eat it too.
image: assets/images/tb.png
---


In today's post I'm going to describe how to set up Tensorboard on google cloud so that you can monitor your deep learning progress with ease.  For the neophytes reading this, Tensorboard is an application that comes with Tensorflow and it allows you to visualize key metrics that are logged while the training is in progress.  The result is served on port 6006 of the 'local' browser and looks something like the picture below.

<img src="https://drive.google.com/uc?id=1gPpKJsJA93PPs-etLmTmhoTo1UyZTSy6" alt="Hope this works!">


In this example, the cross-entropy is displayed.  We can, however, collect and display any statistic from the learning process we want, as well as the graph of the network.  


This is all well and good, but there are a couple of challenges to be met in going from the 'vanilla' instructions on the website to what I actually wanted to do.  First off, I wanted to run the training on google cloud so that my model is working even while my computer (and me) are sleeping.  Second, I'm not using Tensorflow directly, I'm using Pytorch and Tensorflow is merely serving as a backend.  Thus, some of the instructions for setting things up need to be adjusted.

For the second problem, Pytorch vs Tensorflow, there is an off the shelf solution: <a href="https://pypi.org/project/tensorboardX/">TensorboardX</a> or <a href="https://github.com/lanpa/tensorboardX"> here</a> for the source.  This very straight-forward to get the hang of.  Here I'll just give a very simple example of the basic usage.  The procedure is as follows:

1. `pip` install TensorboardX.
2. In your (Python) header include `form tensorboardX import SummaryWriter`
3. Before your training loop, declare a summary writer via:
    `writer=SummaryWriter()`
4. For any variable, `v` which you'd like to log during training, include:  
`writer.add_scalar('logfile',v,i)`, where 'logfile' is the path to the log file that you want to put this information in.  (No harm in literally just using 'logfile'!) and 'i' is the iteration that we are on in the training run.  So, you can think of `i` and `v` as being like the `x` and `y` coordinates of a scatter plot that we are going to make.
5. It's good practice to close the writer after the training loop: `writer.close()`
6. Now, start training!  We will immediately start generating data that gets dumped to the logfile.
7. Now, in the terminal, type:
`tensorboard --logdir path`, where path leads to the directory where `logfile` lives.
8. Great, we have launched the tensorboard!  Where does it live?  Well, this is essentially a web app, so it lives in your browser.  By default the images are served to port 6006.  So, in the browser, just go to `localhost:6006` and see!

So far so good - this is a standard Tensorboard workflow.  However, it is a little more tricky when we want to do the training on a cloud platform like Google Cloud.  There, we can indeed follow all of these steps, but the result will just be displayed for a browser listening to the `localhost` in Google's compute center!  How do we get access to this?

A similar problem already occurs if we want to have *any* interactive GUI environment while working on the cloud.  Sure, we can easily run scripts on the cloud, but how do we see what is going on?  The answer: **port mapping**.  We need to map the output of the `localhost` on Google's machine to the local host on our machine.  Assuming we already have the google compute instance set up and running and the gcloud cli installed, the magic incantation to do port mapping is:

```python
gcloud compute ssh instance_name \
  --project project_id \
  --zone zone_name \
  -- -NL 4321:localhost:6006
```
(See <a href="https://cloud.google.com/solutions/connecting-securely">here</a> for more details.) After running this, we can now just go to our browser and enter the url: `localhost:4321` viola! Tensorboard let's the tensors flow straight to us!  Here is a actual screenshot from my first two (not very succesful) runs. (Note that we need to change the name of the logfile between runs in order to save more than one run):

<img src="https://drive.google.com/uc?id=1LQ_rl7UTwyVE91XXStlH9Obl-Tsp1GgT" alt="tb2">


Again, the port mapping technique is not specific to Tensorboard and is useful for any GUI we want to use on the cloud.  For instance, suppose we are running <a href="http://zeppelin.apache.org/">Zeppelin</a> (highly recommended) on google cloud.  The canonical port is `8080`, so we just need to run:

```python
gcloud compute ssh instance_name \
  --project project_id \
  --zone zone_name \
  -- -NL 1111:localhost:6060
```

Of course, we also need to pick an unoccupied port for our local browser!

Now we are all set!  You can run a massive training run on the cloud all week, check in on it's progress, and not worry about having to babysit your computer :) Hope this was informative.  Now, I'm going to get back to trying to figure out why my training curves are so flat!
