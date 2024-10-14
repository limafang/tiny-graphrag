GEN_NODES = """
## Goal
Please identify and extract triplet information from the provided article, focusing only on entities and relationships related to significant knowledge points. 
Each triplet should be in the form of (Subject, Predicate, Object). 
Follow these guidelines:

1. **Subject:** Concepts in Bayesian Optimization
2. **Predicate:** The action or relationship that links the subject to the object.
3. **Object:** Concepts in Bayesian Optimization that is affected by or related to the action of the subject.

## Example
For the sentence "Gaussian Processes are used to model the objective function in Bayesian Optimization" the triplet would be:

<triplet><subject>Gaussian Processes</subject><predicate>are used to model the objective function in</predicate><object>Bayesian Optimization</object></triplet>

For the sentence "John read a book on the weekend," which is not related to any knowledge points, no triplet should be extracted.

## Instructions
1. Read through the article carefully.
2. Think step by step. Try to find some useful knowledge points from the article. You need to reorganize the content of the sentence into corresponding knowledge points.
3. Identify key sentences that contain relevant triplet information related to significant knowledge points.
4. Extract and format the triplets as per the given example, excluding any information that is not relevant to significant knowledge points.

## Output Format
For each identified triplet, provide:
<triplet><subject>[Entity]</subject><predicate>The action or relationship</predicate><object>The entity</object></triplet>

## Article

{text}

## Your response
"""

GET_ENTITY = """
## Goal

You are an experienced machine learning teacher. 
You need to find out the concepts related to machine learning that the article requires students to master.  

## Example

article:
"In the latest study, we explored the potential of using machine learning algorithms for disease prediction. We used support vector machines (SVM) and random forest algorithms to analyze medical data. The results showed that these models performed well in predicting disease risk through feature selection and cross-validation. In particular, the random forest model showed better performance in dealing with overfitting problems. In addition, we discussed the application of deep learning in medical image analysis."

response:
<concept>Support Vector Machine (SVM)</concept>
<concept>Random Forest Algorithm</concept>
<concept>Feature Selection</concept>
<concept>Overfitting</concept>
<concept>Deep Learning</concept>

## Formate

Wrap these concepts in the HTML tag <concept> and return

## Article

{text}

## Your response
"""

GET_TRIPLETS = """

## Goal
Identify and extract all the relationships between the given concepts from the provided text.
Identify as many relationships between the concepts as possible.
The relationship in the triple should accurately reflect the interaction or connection between the two concepts.

## Guidelines:
1. **Subject:** The first entity from the given entities.
2. **Predicate:** The action or relationship linking the subject to the object.
3. **Object:** The second entity from the given entities.

## Example:
1. Article :
    "Gaussian Processes are used to model the objective function in Bayesian Optimization" 
   Given entities: 
   ["Gaussian Processes", "Bayesian Optimization"].
   Output:
   <triplet><subject>Gaussian Processes</subject><predicate>are used to model the objective function in</predicate><object>Bayesian Optimization</object></triplet>

2. Article :
    "Hydrogen is a colorless, odorless, non-toxic gas and is the lightest and most abundant element in the universe. Oxygen is a gas that supports combustion and is widely present in the Earth's atmosphere. Water is a compound made up of hydrogen and oxygen, with the chemical formula H2O."
    Given entities: 
    ["Hydrogen","Oxygen","Water"]
    Output:
    <triplet><subject>Hydrogen</subject><predicate>is a component of</predicate><object>Water</object></triplet>
3. Article :
    "John read a book on the weekend" 
    Given entities: 
    []
    Output:
    None

## Format:
For each identified triplet, provide:
**the entity should just from "Given Entities"**
<triplet><subject>[Entity]</subject><predicate>[The action or relationship]</predicate><object>[Entity]</object></triplet>

## Given Entities:
{entity}

### Article:
{text}

## Additional Instructions:
- Before giving your response, you should analyze and think about it sentence by sentence.
- Both the subject and object must be selected from the given entities and cannot change their content.
- If no relevant triplet involving both entities is found, no triplet should be extracted.
- If there are similar concepts, please rewrite them into a form that suits our requirements.

## Your response:
"""

TEST_PROMPT = """
## Foundation of students
{state}
## Gole
You will help students solve question through multiple rounds of dialogue.
Please follow the steps below to help students solve the question:
1. Explain the basic knowledge and principles behind the question and make sure the other party understands these basic concepts.
2. Don't give a complete answer directly, but guide the student to think about the key steps of the question.
3. After guiding the student to think, let them try to solve the question by themselves. Give appropriate hints and feedback to help them correct their mistakes and further improve their solutions.
4. Return to TERMINATE after solving the problem
"""

COMMUNITY_PROMPT = """
## Goal

## Example
"""
