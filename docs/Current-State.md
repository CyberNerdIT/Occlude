# OCCLUDE Current Performance

- The video: The Thinking Game from YouTube, 1:24:00 long.
- Has many multi people scenes.
- It is a documentary so it has head-on shots of individuals.
- Many partial hair and wrist (men and woman)


## The Failure Points Observed

### False Positives

- It consistently blurs the main subject Demis Hassabis in his younger ages. The times where he has no
beard or stubble.
- It misses people on edges of the screen. Not partial but fully seen people that is on the right
or left. A clear example is one of the opening sequences where in a broad room a lady holds a phone
talking with AI and she has short hair (not men-like) and jeans, it totally misses it majority of
the scene. It blurs at some point but then loses it again.
- In the same scene there is a man sits on a desk and another man in the background and it blurs all.
My guess is since the face is not visible it plays safe with blur and blurs it anyway. That was the behavior
we prefered but it needs fine-tuning so that it makes sense.
- Another instance of this overdoing is present with scenes that show a lot of people from their backs sitting
like a library or computer lab that show people from behind. Again it sees hair and it tries to blur all and
a muddy frame forms.
- It is late to blur in many cases, for example even with head-on shots we can see the woman for 1 sec and then
it blurs and it is not consistent, it comes and goes so one moment you don't see the woman in the other there
is no blur and you see the woman.
- Kids are still overly blured, we must've refined it before but it might need more tuning.
- It even blocks a computer character where it is not even a human just a dummy human colored form.

## Trajectory 
### Current Model 
It is clear that we have at the limit in terms of the model capability and the way we set rules for the model.
It is expected that the islamic modesty rules with computer vision + blocking accordingly is not a high demand
field so it does not perform the actions accurately, however I think this is not the best version it 
could be. 

### Case of Newer and/or Better Model
Before doing any fine-tuning, we need to make sure that we dont have any other model and/or tool option 
we didnt look at. Becuase having a model that might work well and wiring it is a lot easier than make this model work.
The initial ideas were about some models that could work on the mac mini but because of the nature of the 
task I think we should aim at A100, H100 CUDA territory by default. So if there is a newer or better model that
we didn't choose because old constraints we might want to eval freshly again.

## Final Thoughts

We knew that this will be a long run, but I want to make sure that we are depleting everything before we go on to the next
model/tool.

To restate the most important things as our requirements:

1. Accuracy over Speed (if it will be more accurate but will add an hour more so be it.)
2. If efficiency combats with quality choose quality.
3. Specifically think about edge cases and how to combat them.
4. Weigh options both ways, a change could make a working section worse, always think cohesively.
5. The end result is what matters the most, if it can't be watched comfortably it does not matter how great of a system it is.
6. Always go for the extra mile if you think it is worth it. But before doing anything make sure it is sensible.
7. Research deeply not surface level before attempting to anything or declaring a decision.
8. DON'T USE MORE THAN 5 AGENT. DO NOT LET SUBAGENTS SPAWN THEIR OWN SUBAGENTS.
And now I'll leave it to you. If in any moment you need to ask a question for a decision let me know.
