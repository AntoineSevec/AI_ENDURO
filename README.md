# AI_ENDURO

I can think of other solutions to keep the car in lane and get more points, But first it's important to train the ANN so that it could properly play the game

# Report

* Reaching a human-level performance is obviously a plus, but what is also interesting (and expected) is a meaningful exploration and analysis of the setting that you have chosen.

* Please write down your conclusions as a report with results AND interpretation / explanation.

* Start your report with a general overview of your setting, summary of what has been done, and main conclusions, BEFORE digging into the details.

* Do not forget to write your names in the beginning of your reports / code / notebooks.

## The model

Sequential

The topology :

* 1st layer : A convulutional 2 D layer
* 2nd layer : A Max Pooling 2 D layer
* 3rd layer : Flatten layer
* 4th layer : Dense layer with reLu activation 200 units
* 5th layer : Dense layer with reLu activation 100 units
* 6th layer : Dense layer with sigmoid activation 10 units
* 7th layer : Dense layer with sigmoid activation 10 units
