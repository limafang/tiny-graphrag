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

GEN_COMMUNITY_REPORT = """
## Role
You are an AI assistant that helps a human analyst to perform general information discovery. 
Information discovery is the process of identifying and assessing relevant information associated with certain entities (e.g., organizations and individuals) within a network.

## Goal
Write a comprehensive report of a community.
Given a list of entities that belong to the community as well as their relationships and optional associated claims. The report will be used to inform decision-makers about information associated with the community and their potential impact. 
The content of this report includes an overview of the community's key entities, their legal compliance, technical capabilities, reputation, and noteworthy claims.

## Report Structure

The report should include the following sections:

- TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.
- SUMMARY: An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.
- DETAILED FINDINGS: A list of 5-10 key insights about the community. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below. Be comprehensive.

Return output as a well-formed JSON-formatted string with the following format:
{{
"title": <report_title>,
"summary": <executive_summary>,
"findings": [
{{
"summary":<insight_1_summary>,
"explanation": <insight_1_explanation>
}},
{{
"summary":<insight_2_summary>,
"explanation": <insight_2_explanation>
}}
...
]
}}

## Grounding Rules
Do not include information where the supporting evidence for it is not provided.

## Example Input
-----------
Text:
```
Entities:
```csv
entity,description
VERDANT OASIS PLAZA,Verdant Oasis Plaza is the location of the Unity March
HARMONY ASSEMBLY,Harmony Assembly is an organization that is holding a march at Verdant Oasis Plaza
```
Relationships:
```csv
source,target,description
VERDANT OASIS PLAZA,UNITY MARCH,Verdant Oasis Plaza is the location of the Unity March
VERDANT OASIS PLAZA,HARMONY ASSEMBLY,Harmony Assembly is holding a march at Verdant Oasis Plaza
VERDANT OASIS PLAZA,UNITY MARCH,The Unity March is taking place at Verdant Oasis Plaza
VERDANT OASIS PLAZA,TRIBUNE SPOTLIGHT,Tribune Spotlight is reporting on the Unity march taking place at Verdant Oasis Plaza
VERDANT OASIS PLAZA,BAILEY ASADI,Bailey Asadi is speaking at Verdant Oasis Plaza about the march
HARMONY ASSEMBLY,UNITY MARCH,Harmony Assembly is organizing the Unity March
```
```
Output:
{{
"title": "Verdant Oasis Plaza and Unity March",
"summary": "The community revolves around the Verdant Oasis Plaza, which is the location of the Unity March. The plaza has relationships with the Harmony Assembly, Unity March, and Tribune Spotlight, all of which are associated with the march event.",
"findings": [
{{
"summary": "Verdant Oasis Plaza as the central location",
"explanation": "Verdant Oasis Plaza is the central entity in this community, serving as the location for the Unity March. This plaza is the common link between all other entities, suggesting its significance in the community. The plaza's association with the march could potentially lead to issues such as public disorder or conflict, depending on the nature of the march and the reactions it provokes."
}},
{{
"summary": "Harmony Assembly's role in the community",
"explanation": "Harmony Assembly is another key entity in this community, being the organizer of the march at Verdant Oasis Plaza. The nature of Harmony Assembly and its march could be a potential source of threat, depending on their objectives and the reactions they provoke. The relationship between Harmony Assembly and the plaza is crucial in understanding the dynamics of this community."
}},
{{
"summary": "Unity March as a significant event",
"explanation": "The Unity March is a significant event taking place at Verdant Oasis Plaza. This event is a key factor in the community's dynamics and could be a potential source of threat, depending on the nature of the march and the reactions it provokes. The relationship between the march and the plaza is crucial in understanding the dynamics of this community."
}},
{{
"summary": "Role of Tribune Spotlight",
"explanation": "Tribune Spotlight is reporting on the Unity March taking place in Verdant Oasis Plaza. This suggests that the event has attracted media attention, which could amplify its impact on the community. The role of Tribune Spotlight could be significant in shaping public perception of the event and the entities involved."
}}
]
}}

## Real Data
Use the following text for your answer. Do not make anything up in your answer.

Text:
```
{input_text}
```

The report should include the following sections:

- TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.
- SUMMARY: An executive summary of the community's overall structure, how its entities are related to each other, and significant information associated with its entities.
- DETAILED FINDINGS: A list of 5-10 key insights about the community. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below. Be comprehensive.

Return output as a well-formed JSON-formatted string with the following format:
{{
"title": <report_title>,
"summary": <executive_summary>,
"rating": <impact_severity_rating>,
"rating_explanation": <rating_explanation>,
"findings": [
{{
"summary":<insight_1_summary>,
"explanation": <insight_1_explanation>
}},
{{
"summary":<insight_2_summary>,
"explanation": <insight_2_explanation>
}}
...
]
}}

## Grounding Rules
Do not include information where the supporting evidence for it is not provided.

Output:
"""
