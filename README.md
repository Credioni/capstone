# Project Schedule

| First Week    | Second Week   | Third Week   | Fourth Week |
| ------------- | ------------- |------------- |-------------|
| Content Cell  | Content       | Content      |  Content    |


# Project 19: Trustworthy AI-Generated Code via Inductive and Deductive Reasoning

AI-generated code lacks formal correctness guarantees, which can make it unreliable for critical applications. This project combines LLMs (**inductive models**) with formal verification tools (**deductive models**) to assess and improve AI-generated code quality.\
The focus will be on automating code correctness checks through empirical analysis and formal verification methods.
<br/><br/>
**Task**
<br/>
Develop a lightweight AI-assisted coding assistant that generates Python or Java code and verifies
its correctness using logical reasoning. This system will be useful for educational tools, secure
software development, and automated bug detection.
<br/><br/>

# How to start?
1. Train a baseline image retrieval model using OpenAI CLIP.
2. Use CodeLlama or StarCoder to generate Python or Java functions based on user prompts.
3. Use Z3 theorem prover to check the logical correctness of AI-generated code.
4. Implement static analysis tools (PyLint, SonarQube) to identify errors and code inefficiencies.
5. Build a Streamlit or VS Code Extension where users input code prompts and see both AIgenerated code and its verification results.
6. Compare verified vs. unverified AI-generated code, measuring error detection rate and
efficiency.
7. Adjust AI prompts and verification rules to improve correctness detection.


# Approach
 - Should we use python? Since their compiler produces errors in **runtime**? Using statically typed languages would be easier to debug.
 - Java, Rust?



## Fine-Tuning & Datasets
 - LLM DeepSeek-R1
 - Datasets to be used
## Formal Verification Tools
## Empirical Testing