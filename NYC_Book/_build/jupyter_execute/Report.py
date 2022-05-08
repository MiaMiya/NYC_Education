#!/usr/bin/env python
# coding: utf-8

# # Introduction 

# "Achieving inclusive and quality education for all reaffirms the belief that education is one of the most powerful and proven vehicles for sustainable development." - [UN](https://www.undp.org/sustainable-development-goals) 
# 
# It should be the goal of any governing organization to ensure high-quality education for all, as its benefits are high and extensive. From the fact that illiteracy means you have a substantially higher likelihood of ending up in jail or on welfare, that illiteracy has a negative impact on discrimination and preventable diseases, or the fact that for every dollar spent on adult illiteracy the ROI is 6.14$ (614%). Another extremely important effect of education is the social network you get, which combats loneliness which in itself has an array of negative health impacts [2](https://www.hrsa.gov/enews/past-issues/2019/january-17/loneliness-epidemic).  
# Given that there is no doubt about the importance of education it's important to investigate when the educational system fails and people drop out ?!?Nyt ord!?!, and which factors have an impact on the dropout?!?Nyt ord!?!. To investigate this we'll look at poverty data from New York City in 2015 ?!?!?!?!?!?. 

# ## The Data
# The data can be obtained from [data.cityofnewyork](https://data.cityofnewyork.us/City-Government/NYCgov-Poverty-Measure-Data-2015-/ahcp-vck8).  
# It contains 69103 observations and 61 columns of which we only use a subset of ?!?!?! columns:
# 

# ### Exploratory Data analysis

# ## An investegation of the impact of education on key attributes
# 

# ## Salary
# An obvious attribute that we would expect education to have an impact on is income. It is intuitive that better education leads to better and more well-paid jobs, so let's investigate this claim. We'll do this by looking at the distribution of income created from the combined !="!="#€)!"=€)!="€) attributes:
# 
# 
# !?!"?WEGJWORJAFOPJPOTJEPOJSPOJSGPOJSDsoighsdobusdgijah PLot her iuhausyhf98jw   brxa0jsiebsgiuhsdfsd9hriwepqhgiughsagilugsdaguahdgiuhdaighdsuhgiusdhghsdohsoighusgnsoungusdbgo
# 
# What we see is indeed that you can expect a higher salary given you've attained a higher education level. This can be seen in both mean, median, and max income. Thus it's easy to conclude that education is an effective tool against poverty, however, it's important to note that we know nothing of the jobs that people occupy, so a higher salary does not necessarily mean a job that is a "vehicle for sustainable development". 

# ## Sex a hopeful story
# SDG 4 says: "Ensure inclusive and equitable quality education and promote lifelong learning opportunities for all" [3](https://sdgs.un.org/goals/goal4) this unsurprisingly also includes women. Worldwide we know there is a discrepancy between males and females from Hans Rosling's quiz in the opening of his famous book Factfullness: "Worldwide, 30-year-old men have spent 10 years in school, on average. How many years have women of the same age spent in school?" the answer is 9 years [4](https://factfulnessquiz.com). This number is of course or US or NYC specific, thus let's take a look: 
# 
# 
# It's fairly clear that there is no difference in education obtained between the sexes, however, let's also see if this holds true by race:
# ### Race and Sex
# 
# 
# ### Salary and Sex a sad story
# Although the equality of education between the sexes is a good sign, it's an entirely different and alarming story when looking at Sex and salary: 
#     porjasporjaspiorjaspojrpoasjrfpoasjfpojaposajfpjasgihsdigasohisjg
# Here we have a fairly big discrepancy. This is very alarming as it contradicts our previous conclusion: that higher education means a higher salary and thus an effective tool against poverty. The two plots above suggest that although females have an equal amount of education as men, they still have a lower average salary, and thus a higher likelihood of being in poverty. Now you might think that this could be explained by "Stay-at-home-momes" but there is a difference in salary between men and women 120+4i12+0i4+012+04912+09. Thus it would be interesting to investigate the effect of **Sex** on poverty. 
# 
# ### Age and race
# Looking at education we would expect it to be higher the lower the age (age > 24). Generally, we expect people to be more educated now than 40 years ago. Additionally, the American society is generally less segregated, especially compared to say the 60s, thus we would expect to see a greater increase in education for all other races. 

# # Creating a predictive model that should be bad a predictor (but isn't)

# iashafhafohasfoih we'll look at predicting the attained education level given attributes that shouldn't really have an impact on education level. These attributes are: **Age**, **Borough** (location), **Disability**, **languages other than English spoken**, **number of people in the household**, **Race**, **Sex**.   
# Interestingly we would expect **languages other than English spoken** to have a negative impact on the level of education attained, as the majority of Americans only speak English, thus speaking another language than English is an indication the person is of another race. Of course, attributes like **Race** and **Sex** having an influence on education would go directly against SDGs 4, and 5. 
# Finally, looking at the heat maps !?!??!!? we see that there are concentrations of ?!?!?!?! ?in =!=!=!=!=!, thus **Borough** having an impact on education, is another way of saying **Race** has an impact. Additionally, we know that a community (**Borough**) has a positive feedback loop in either direction, so higher education leads to even higher education and vice versa !)!))")#=)!"=)#="! see heat map !="?!"?#=!"?€=!"€=€!"?=€!"€?=. 

# 
