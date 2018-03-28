
# Research Outline

## Ingredients

 - Optimization of scheduling based on model of learner's activation decay
 - Further optimization of activation by tracking response times
 - Presentation of words in-context by generating multiple-choice fill-in-the-blank questions
 - Implementing scaffolding to improve learning experience
 - Providing more meaningful feedback to the user
 - Including other proven learning methodologies
 - Learning effects described by Dr. Ramscar

[TOC]


### Scheduling

*Folder of papers: [link][1]*

\[x][[link][2]] **Sense, F. et al.** - Stability of Individual Parameters in a Model of Optimal Fact Learning 
> - Includes formulas for activation and decay
> - Includes flowchart for ideal item selection

\[x][[link][3]] **Sense, F. et al.** - An Individual's Rate of Forgetting Is Stable Over Time but Differs Across Materials 
> - Cites many articles describing the success of the spacing and testing effects (p2-3)
> - Is very similar to above article


\[x][[link][4]] **Koelewijn, L.** - Optimizing Fact Learning Gains 
> - Cites many articles establishing the background of learning, and computer learning specifically (p5)
> - Is very similar to the van Thiel paper
> - Tests the user's reaction time right away to determine baseline reaction time value individually
> - Also uses the median of all initial trials as an indicator of the initial alpha value, rather than setting a fixed alpha (p13), but has middling results (p33).
> - Compares directly with the van Thiel paper
> - Contains both laboratory and real-world experiments and results
> - In addition to using initial tested reaction time, base reaction time value (f) can also be optimized for by finding the greatest standard deviation of activation (alpha). (p21)
> - Warns about overlearning (p36)

\[x][[link][5]] **van Rijn, H. et al.** - Passing the Test:  Improving Learning gains by Balancing Spacing and Testing Effects
> - Claims that the Pavlik/Anderson model is based on the work of Anderson/Schooler (1991)
> - Describes using new encounters to adjust activation by a fixed amount in the correct direction, rather than doing a hard recalculation, due to noise being large.  (p3-4 - Subsymbolic Model Tracing)
> - Only adjusts activation based on reaction time if the difference between expected and observed reaction time is >0.5s (p4 - Algorithm 4)

\[x][[link][6]] **Pavlik & Anderson (2005)** - Practice and Forgetting Effects on Vocabulary Memory:
An Activation-Based Model of the Spacing Effect
> - Paper that serves as basis for most other papers linked here for learning scheduling
> - Mentions other possible models for modeling learning/forgetting other than the power-based function used by all of these papers, and provides their rationale for choosing a power function (p3)
> - Contains interesting figures demonstrating the spacing effect (p7)
> - Describes a formula for assigning a probability of recall based on activation (p10)
> - Describes scaling of the passage of time outside an experiment as "psychological time" (p10-11)
> - Theorizes about biological explanations for the observed learning/forgetting processes

\[x][[link][7]] **Pavlik & Anderson (2008)** - Using a Model to Compute the Optimal Schedule of Practice
> - Serves as basis for most other papers linked here for learning scheduling
> - Contents are mostly covered by other papers that heavily refer to it


\[x][[link][8]] **Settles, B. and Meeder, B.** - A Trainable Spaced Repetition Model for Language Learning
> - Paper from Duolingo
> - Contains a different method of determining the probability of recall for a word as a function of time
> - Serves as a look into a real-world application of these methods on a large scale
> - Includes link to code (p9)

\[x][[link][9]] **Lindsey, V. et al** - Improving Students' Long-Term Knowledge Retention Through Personalized Review
> - Doesn't contain much useful information not otherwise discussed in other papers

\[x][[link][10]] **Streeter, M.** - Mixture Modeling of Individual Learning Curves
> - Paper from Duolingo
> - Really dense math, difficult to follow

[[link][11]] **Lindsey, R. et al.** - Optimizing Retention with Cognitive Models
> - Example for balancing spacing and testing effects

[[link][12]] **Anderson, J. and Schooler, L.** - Reflections of the Environment in Memory
> - Basis for Pavlik & Anderson models

[[link][13]] **Raaijmakers, J.** - Spacing and repetition effects in human memory:  application of the SAM model
> - Describes the SAM model

[[link][14]] **Anderson, J. et al.** - An Integrated Theory of the Mind
> - Source for describing the "ACT-R cognitive modelling architecture"

[[link][15]] **Pavlik, P.** - Understanding and applying the dynamics of test practice and study practice
> - Source for beta-parameter extension to activation formula

[[link][16]] **Nijboer, M.** - Optimal Fact Learning:  Applying Presentation Scheduling to Realistic Conditions
> - Masters thesis
> - Contains good explanation of how to adjust alpha values based on reaction times (p20-21)
> - Contains graphs that demonstrate how spacing affects activation in ACT-R (p18)
> - Contains formulas for estimated reading times of sentences (p23, p51)
> - Describes improvements to the "psychological time" explanation for a slowed rate of forgetting between sessions (p27-28)

[[link][17]] **van Woudenberg, M.** - Optimal Word Pair Learning in the Short Term:  Using an Activation Based Spacing Model
> - Uses very basic formula for adjusting alpha based on reaction times


---

### Reaction Timing

*Folder of papers: [link][21]*

\[x][[link][22]] **van Thiel, W.** - Optimize Learning with Reaction Time-based Spacing 
> - Contains a lot of formulas for optimizing predicted activation based on reaction times
> - Serves as a strong basis for any experiments employing the measuring of reaction times

---

### Vocab Learning

*Folder of papers: [link][31]*

\[x][[link][32]] **Gu, Y. and Johnson, R.** - Vocabulary Learning Strategies and Language Learning Outcomes
> - Contains a survey of Chinese students learning English, asking about their vocabulary studying habits, correlating their responses to their test results and vocabulary sizes
> - Indicates which techniques are most well-correlated with success (p26)
>  - Most important:  Self-initiation and context

\[x][[link][33]] **Oxford, R. and Scarcella, R.** - Second Language Vocabulary Learning Among Adults:  State of hte Art in Vocabulary Instruction
> - Provides guidance on effective vocabulary learning strategies, but doesn't offer many sources for their assertions
> - Suggests that it is helpful to learn "themes" or "semantic groupings" of words together, but should be cautious to not teach words that are TOO related together (p9)
>  - Antonyms often appear in the same contexts, and can confound their acquisition if learned simultaneously

[[link][34]] **Atkinson, R. and Paulson, J.** - An approach to the Psychology of Instruction

[[link][35]] **Atkinson, R.** - Optimizing the Learning of a Second-Language Vocabulary

---

### Context Presentation

*Folder of papers: [link][41]*

\[x][[link][42]] **Knoop, S. and Wilske, S.** - Wordgap: Automatic Generation of Gap-Filling Vocabulary Exercises for Mobile Learning
> - Provides succinct explanation on how to implement generation of cloze exercises.

\[x][[link][43]] **Knoop, S.** - Automatic Generation of Multiple-Choice Cloze Exercises for the Android Smartphone 
> - Example bachelor's thesis
> - Cites papers discussing general vocabulary acquisition (p7)
> - Cites papers discussing learning vocabulary in context (p7)
> - Cites papers discussing benefits of cloze exercises (p8)
> - Discusses good distractors in cloze exercises (p9)
> - Lists previous related work (p21)
> - Lists useful libraries/tools (p25)

[[link][44]] **Prince, P.** - Second Language Vocabulary Learning:  The Role of Context versus Translations as a Function of Proficiency

[[link][45]] **Nagy, W.** - On the Role of Context in First- and Second-Language Vocabulary Learning

---

### Scaffolding

*Folder of papers: [link][51]*

*More papers: [link](http://link.springer.com/article/10.1007%2Fs11423-007-9064-3?LI=true)*

*About scaffolding: [link](http://www.tllg.unisa.edu.au/lib_guide/gllt_ch3_sec6.pdf)*

\[x][[link][52]] **Azevedo, et al.** - Adaptive human scaffolding facilitates adolescents’ self-regulated learning with hypermedia
> - Provides references to several types of scaffolding (p6)
> - Describes scaffolding in an experiment, but uses a human tutor to accomplish this.  As a result, this paper may be of limited usefulness

\[x][[link][53]] **Wood, D.** - Scaffolding, Contingent Tutoring and Computer-supported Learning
> - References papers that have applied scaffolding in varied contexts (p2)
> - Doesn't seem to provide any details about implementation, and mostly seems useless

---

### Feedback

*Folder of papers: [link][61]*

\[x][[link][62]] **Narciss, S. and Huth, K** - How to design informative tutoring feedback for multi-media learning
> - Discusses value of feedback in a learning system, but cautions that simply adding more and more feedback does not necessarily lead to better outcomes (p3)
> - References detailed description of 'selection and specification of informative feedback content' (p6)
> - Lists criteria for effect feedback, including: "Do not immediately combine elaborated feedback components with the correct response or solution" (p8-9)
> - Serves as a good source for how to think about planning the feedback procedure

\[x][[link][63]] **Clariana, Roy** - A review of multiple-try feedback
> - Refers to multiple choice questions and the power of multiple-try feedback for them (p7)
> 

---

### Ramscar

*Folder of papers: [link][81]*

\[x][[link][82]] **Dye, Milin, Futrell, & Ramscar** - Cute Little Puppies and Nice Cold Beers:  An Information Theoretic Analysis of Prenominal Adjectives

> - Suggests that it may improve learning to utilize appropriate adjectives in front of nouns to learn
> - May also suggest beyond nouns that looking for common preceding words may help learning of succeeding words
> - Very new paper, may have to inquire about citing it

[[link][83]] **Arnon, I. and Ramscar, M.** - Granularity and the acquisition of grammatical gender:  How order-of-acquisition affects what gets learned

[[link][84]] **Dye, Milin, Futrell, & Ramscar** - A Functional Theory of Gender Paradigms

[[link][85]] **Ramscar, M. and Dye, M. and McCauley, S.** - Expectation and error distribution in language learning:  The curious absence of "mouses" in adult speech

[[link][86]] **Ramscar, M. et al.** - The effects of Feature-Label-Order and Their Implications for Symbolic Learning

---

### ICALL

[[link][102]] **Grace, C.** - Retention of Word Meanings Inferred from Context and Sentence-Level Translations: Implications for the Design of Beginning-Level CALL Software
> - Describes a system that combines presenting words in context to allow users to infer their meaning, along with L1 translations for verification
> - Has good results

\[x][[link][103]] **Meurers, D. and Amaral, A.** - On Using Intelligent Computer-Assisted Language Learning in Real-Life Foreign Language Teaching and Learning
> - Discusses many common problems with language-learning systems
> - Demonstrates a full-fledged language application

[[link][104]] **Graeser, A., et al.** - Intelligent Tutoring Systems with Conversational Dialogue

[[link][105]] **Self, J.** - Bypassing the Intractable Problem of Student Modelling
> - Suggests that building a student model is of great importance in developing computer tutoring systems

[[link][106]] **Chen, C., Lee, H., and Chen, Y** - Personalized e-learning system using Item Response Theory

[[link][107]] **Nerbonne, J.** - Computer-assisted Language Learning and Natural Language Processing

[[link][108]] **Segler, T., Pain, H., and Sorace, A.** - Second Language Vocabulary Acquisition and Learning Strategies in ICALL Environments
> - Vocab learning ICALL example

---

### Spacing Effect

*Folder of papers: [link][131]*

[[link][132]] **Ebbinghaus, H.** - Memory:  A Contribution to Experimental Psychology
> - Original paper describing the spacing effect

[[link][133]] **Bahrick, P. et al.** - Maintenance of Foreign Language Vocabulary and the Spacing Effect 

[[link][134]] **Janiszewski, C. and Noel, H.** - A Meta-Analysis of the Spacing Effect in Verbal Learning:  Implications for Research on Advertising Repetition and Consumer
> - Spacing effect in advertising

[[link][135]] **Bjork, R. and Allen, T.** - The Spacing Effect:  Consolidation or Differential Encoding?
> - Suggests that differential encoding is the likely driver of the spacing effect

[[link][136]] **Bahrick, H. and Hall, L.** - The importance of retrieval failures to long-term retention:  A metacognitive explanation of the spacing effect
> - Argues for encoding theory of explaining spacing effect

[[link][137]] **Glenberg, A.** - Component-levels theory of the effects of spacing of repetitions on recall and recognition

[[link][138]] **Bloom, K. and Shuell, T.** - Effects of Massed and Distributed Practice on the Learning and Retention of Second-Language Vocabulary
> - Example of spacing effect in vocab learning

[[link][139]] **Rohrer, D. and Taylor, K.** - The Effects of Overlearning and Distributed Practise on the Retention of Mathematics Knowledge
> - Example of spacing effect in math learning

[[link][140]] **Greeno, J.** - Conservation of Information-Processing Capacity in Paired-Associate Memorizing
> - Discusses "diminished processing" theory of spacing effect

[[link][141]] **Hintzman, D. et al.** - Voluntary attention and the spacing effect
> - Discusses "diminished processing" theory of spacing effect

[[link][142]] **Melton, A.** - The Situation with Respect to the Spacing of Repetitions and Memory
> - Discusses "variable encoding" theory of spacing effect

[[link][143]] **Cepeda, N. et al.** - Optimizing Distributed Practice:  Theoretical Analysis and Practical Implications
> - Provides evidence that too much spacing has negative consequences

---

### Machine Learning

*Folder of papers: [link][161]*

[[link][162]] **James, G. et al.** - An Introduction to Statistical Learning, with Applications in R
> - Basics of machine/statistical learning

[[link][163]] **MacKay, D.** - Information Theory, Inference, and Learning Algorithms

[[link][164]] **Hastie, T., Tibshirani, R., and Friedman, J.** - The Elements of Statistical Learning:  Data Mining, Inference, and Prediction

**Blogs:**
[[link][165]] Neural networks and deep learning


---

### Miscellaneous

*Folder of papers:  [link][201]*

\[x][[link][202]] **Gais, Lucas, and Born** - Sleep after learning aids memory recall
> - Provides evidence of the positive effects of sleep more closely following a learning session

[[link][203]] **Rescorla, R. and Wagner, A.** - A Theory of Pavlovian Conditioning:  Variations in the Effectiveness of Reinforcement and Nonreinforcement

[[link][204]] **Carrier, M. and Pashler, H.** - The influence of retrieval on retention

[[link][205]] **Ullman, M.** - Contributions of memory circuits to language:  the declarative/procedural model
> - Explains declarative and procedural memory

[[link][206]] **Anderson, J., Fincham, J., and Douglass, S.** - Practice and Retention:  A Unifying Analysis
> - Claims that the rate of forgetting slows between study sessions



[1]: https://drive.google.com/drive/folders/0B_kzRS5tOgsXeGQ2Wmt0bDliU2s?usp=sharing
[2]: https://drive.google.com/file/d/0B_kzRS5tOgsXUkFKWUNVNVptVXM/view?usp=sharing
[3]: https://drive.google.com/file/d/0B_kzRS5tOgsXMjJVbXdGaVJuc3c/view?usp=sharing
[4]: https://drive.google.com/file/d/0B_kzRS5tOgsXNjdHbHE5QUtOeU0/view?usp=sharing
[5]: https://drive.google.com/file/d/0B_kzRS5tOgsXWmNFNmI5WmwtV3c/view?usp=sharing
[6]: https://drive.google.com/file/d/0B8zQ4O1-JvDrRXFjcmRsVVZYT2s/view?usp=sharing
[7]: https://drive.google.com/file/d/0B8zQ4O1-JvDrYWJ4U21lLWR1MjQ/view?usp=sharing
[8]: https://drive.google.com/file/d/0B8zQ4O1-JvDrbEhIbmdWTHF3VzA/view?usp=sharing
[9]: https://drive.google.com/file/d/0B8zQ4O1-JvDrY01ZbmRPS01qV3c/view?usp=sharing
[10]: https://drive.google.com/file/d/0B8zQ4O1-JvDrWFB4aTBnakRwMjQ/view?usp=sharing
[11]: https://drive.google.com/file/d/0B8zQ4O1-JvDrLVRiQXdFMDQwUkU/view?usp=sharing
[12]: https://drive.google.com/file/d/0B8zQ4O1-JvDrVGROTUluUDRRQlE/view?usp=sharing
[13]: https://drive.google.com/file/d/0B8zQ4O1-JvDrVzBhN1U0bzVqMlk/view?usp=sharing
[14]: https://drive.google.com/file/d/0B8zQ4O1-JvDraGphNEdVYmZOTU0/view?usp=sharing
[15]: https://drive.google.com/file/d/0B8zQ4O1-JvDrWV85QnVvMm5LWms/view?usp=sharing
[16]: https://drive.google.com/file/d/0B8zQ4O1-JvDrYTFVTlJkRW1pYTg/view?usp=sharing
[17]: https://drive.google.com/file/d/0B8zQ4O1-JvDrZEFYbFdrU2MtODA/view?usp=sharing

[21]: https://drive.google.com/drive/folders/0B_kzRS5tOgsXQkJ3am8xakRpdFk?usp=sharing
[22]: https://drive.google.com/file/d/0B_kzRS5tOgsXZG9GbmNVa3llSG8/view?usp=sharing

[31]: https://drive.google.com/drive/folders/0B8zQ4O1-JvDrOUhnS1Zmd1RzYXc?usp=sharing
[32]: https://drive.google.com/file/d/0B8zQ4O1-JvDrTnNSU1pYd04wdU0/view?usp=sharing
[33]: https://drive.google.com/file/d/0B8zQ4O1-JvDrM3psZXlHOGNUTVU/view?usp=sharing
[34]: https://drive.google.com/file/d/0B8zQ4O1-JvDrZk5hdVVmX2Y2NUU/view?usp=sharing
[35]: https://drive.google.com/file/d/0B8zQ4O1-JvDrSXNiNjQxeEhKemM/view?usp=sharing

[41]: https://drive.google.com/drive/folders/0B_kzRS5tOgsXYVA4QzZrTkF2eW8?usp=sharing
[42]: https://drive.google.com/file/d/0B_kzRS5tOgsXZmZvalhVcWx3ODQ/view?usp=sharing
[43]: https://drive.google.com/file/d/0B_kzRS5tOgsXYlFXbXBZOVpEQzg/view?usp=sharing
[44]: https://drive.google.com/file/d/0B8zQ4O1-JvDrZVNiZm1aNllKSk0/view?usp=sharing
[45]: https://drive.google.com/file/d/0B8zQ4O1-JvDrU0ExbUtJdEowNHM/view?usp=sharing

[51]: https://drive.google.com/drive/folders/0B8zQ4O1-JvDramZOWmNnalgxLTQ?usp=sharing
[52]: https://drive.google.com/file/d/0B8zQ4O1-JvDrWk9IaU1aS0dzRkU/view?usp=sharing
[53]: https://drive.google.com/file/d/0B8zQ4O1-JvDrSHJHd3BhOEthYjA/view?usp=sharing

[61]: https://drive.google.com/drive/folders/0B8zQ4O1-JvDrYVVJUC14WWF0NGs?usp=sharing
[62]: https://drive.google.com/file/d/0B8zQ4O1-JvDrTjZsaHY3T1g3bkk/view?usp=sharing
[63]: https://drive.google.com/file/d/0B8zQ4O1-JvDrQll6R2o3U0x5YjQ/view?usp=sharing

[81]: https://drive.google.com/drive/folders/0B8zQ4O1-JvDrQzM3QV9fY21FZUk?usp=sharing
[82]: https://drive.google.com/file/d/0B8zQ4O1-JvDrbWRybHp0dU5KanM/view?usp=sharing
[83]: https://drive.google.com/file/d/0B8zQ4O1-JvDrS25iOFFQUGMweEU/view?usp=sharing
[84]: https://drive.google.com/file/d/0B8zQ4O1-JvDrSExTT2JaNDE0dDA/view?usp=sharing
[85]: https://drive.google.com/file/d/0B8zQ4O1-JvDrcHJDai12aVlOX28/view?usp=sharing
[86]: https://drive.google.com/file/d/0B8zQ4O1-JvDrWE9EN2lwS196NWM/view?usp=sharing

[101]: https://drive.google.com/drive/folders/0B8zQ4O1-JvDrZDRhMkdaOXBzM0U?usp=sharing
[102]: https://drive.google.com/file/d/0B8zQ4O1-JvDrbGJIaE5tM2Z1NW8/view?usp=sharing
[103]: https://drive.google.com/file/d/0B8zQ4O1-JvDrdTdfamVFT3FQclU/view?usp=sharing
[104]: https://drive.google.com/file/d/0B8zQ4O1-JvDrLXkyM09YOV9hVUU/view?usp=sharing
[105]: https://drive.google.com/file/d/0B8zQ4O1-JvDramEwcGxLZHFKbnM/view?usp=sharing
[106]: https://drive.google.com/file/d/0B8zQ4O1-JvDrT1MyYlVFTDR5UzQ/view?usp=sharing
[107]: https://drive.google.com/file/d/0B8zQ4O1-JvDremJVUmdsb3FlVTA/view?usp=sharing
[108]: https://drive.google.com/file/d/0B8zQ4O1-JvDra3NXWFhMSlE1dU0/view?usp=sharing

[131]: https://drive.google.com/drive/folders/0B8zQ4O1-JvDrVDQ1SGdacXllZXM?usp=sharing
[132]: https://drive.google.com/file/d/0B8zQ4O1-JvDrWVhMMDltaFBPam8/view?usp=sharing
[133]: https://drive.google.com/file/d/0B8zQ4O1-JvDrcmRxM2xYZVpjak0/view?usp=sharing
[134]: https://drive.google.com/file/d/0B8zQ4O1-JvDrOHR3b0xQa1lyZzQ/view?usp=sharing
[135]: https://drive.google.com/file/d/0B8zQ4O1-JvDrem1tcHBmOUtBa3M/view?usp=sharing
[136]: https://drive.google.com/file/d/0B8zQ4O1-JvDrRGZ4Y1BfT25LWFE/view?usp=sharing
[137]: https://drive.google.com/file/d/0B8zQ4O1-JvDrY1htMjFtTGNlbm8/view?usp=sharing
[138]: https://drive.google.com/file/d/0B8zQ4O1-JvDrclVPSHFFaWxnR28/view?usp=sharing
[139]: https://drive.google.com/file/d/0B8zQ4O1-JvDrMFNZajFhY3BBQTQ/view?usp=sharing
[140]: https://drive.google.com/file/d/0B8zQ4O1-JvDrQjVOQ29mQlhDWDg/view?usp=sharing
[141]: https://drive.google.com/file/d/0B8zQ4O1-JvDrN1cxcWtNMEZTWnc/view?usp=sharing
[142]: https://drive.google.com/file/d/0B8zQ4O1-JvDrblMyQmlWWFdQTlk/view?usp=sharing
[143]: https://drive.google.com/file/d/0B8zQ4O1-JvDrdFBOdVRQaENLSEU/view?usp=sharing

[161]: https://drive.google.com/drive/folders/0B8zQ4O1-JvDrQ3pLYzVHS3l5cnc?usp=sharing
[162]: https://drive.google.com/file/d/0B8zQ4O1-JvDranNzWi1pbjJ4WUU/view?usp=sharing
[163]: https://drive.google.com/file/d/0B8zQ4O1-JvDrd2NhaDNkVUx5SGs/view?usp=sharing
[164]: https://drive.google.com/file/d/0B8zQ4O1-JvDralpQanZCOHVWalE/view?usp=sharing
[165]: http://neuralnetworksanddeeplearning.com/chap1.html

[201]: https://drive.google.com/drive/folders/0B8zQ4O1-JvDrakZSWFpySU1VaDg?usp=sharing
[202]: https://drive.google.com/file/d/0B8zQ4O1-JvDrSXFKSGtTVE1WMmM/view?usp=sharing
[203]: https://drive.google.com/file/d/0B8zQ4O1-JvDrSFRRa19xR0NCMWc/view?usp=sharing
[204]: https://drive.google.com/file/d/0B8zQ4O1-JvDrbnRRblRac1djX2M/view?usp=sharing
[205]: https://drive.google.com/file/d/0B8zQ4O1-JvDrYzdRS1RWWTFEWFk/view?usp=sharing
[206]: https://drive.google.com/file/d/0B8zQ4O1-JvDrdHdsVGRVdDNyMEU/view?usp=sharing

---

> Written with [StackEdit](https://stackedit.io/).
