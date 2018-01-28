Almost every individual has a credit card and a correct usage of it does really help in building a good credit score. 
How important is Credit score? Ask anyone – specially here in US and all will know it. However, one default payment 
can cause the credit score to plummet. Not only credit score, it also has an adverse effect on the credit limit and 
future loans of any kind.
In this project, we would be dealing with a case of customer default payments in Taiwan. The dataset has 24 features and 
a class label and there are 30000 instances. We would be creating a predictive model to let the bank predict that whether 
their customer would be a defaulter for the next payment or not.

Objective: To build classifiers and use them to predict whether a credit card customer will be a defaulter in his next payment. 
Dataset: Data: Bank Dataset from UCI Repository https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients#

Before we start our analysis, we need to get familiarized with the data set. The dataset is an imbalanced one i.e. majority of the class labels are non-defaulters. Out of 30000 instances, 78% are non-defaulters and remaining 22% are defaulters. 
Following are the features present in the dataset. ID, Credit balance, Gender, Education, marital status and age are self-explanatory. Pay_0, Pay_2, Pay_3, Pay_4, Pay_5, Pay_6 are repayment status of months April till September respectively. Repayment status is defined as the delay in payment. Example: if the value of Pay_0 is -1 then it means that the customer has duly paid, if the value is 2 then it means that the payment is overdue for two months. Bill_Amt1 to Bill_Amt6 are the credit card bill amounts for the month of April till September. Pay_Amt1 to Pay_Amt6 are the amount that the customer has paid against the credit card bill from the month of April till September. 
N.B. The data belongs to year 2005.

The features are listed down in a tabular form for a better understanding.
Attribute	          Explanation
ID	                Unique Id of each record
Credit Balance	        Credit amount on the given credit card
Gender	                Gender of the customer i.e. male or female
Education	        Education level of the customer i.e. high school, graduate, university, others
Marital Status	        Marital status i.e. married, single, others
Age	                Age of the customer
Pay_0	                Repayment status in September
Pay_2	                Repayment status in August
Pay_3              	Repayment status in July
Pay_4	                Repayment status in June
Pay_5	                Repayment status in May
Pay_6	                Repayment status in April
Bill_Amt1	        Bill Amount in September
Bill_Amt2	        Bill Amount in August
Bill_Amt3	        Bill Amount in July
Bill_Amt4	        Bill Amount in June
Bill_Amt5	        Bill Amount in May
Bill_Amt6	        Bill Amount in April
Pay_Amt1	        Amount paid in September
Pay_Amt2	        Amount paid in August
Pay_Amt3	        Amount paid in July
Pay_Amt4	        Amount paid in June
Pay_Amt5	        Amount paid in May
Pay_Amt6	        Amount paid in April
Default_payment_next_month	Either 0 or 1 -  0 means for the next payment the customer is not a defaulter and 1 means the customer is a defaulter

We have used following classifiers : C5.0, c5.0Cost,rpart,random forest

Comparative study of all four Classifiers
	            C5.0	    C5.0Cost	    Rpart	  Random Forest
Accuracy	    0.8183	   0.8165	      0.8144	   0.8138
Sensitivity	  0.32792	   0.31694	    0.30295  	 0.35763
Specificity	  0.95889	   0.95975	    0.96105	   0.94469
Precision	    0.69586	   0.69313	    0.69052	   0.64968

As per this business problem, a classifier would be considered better if it has a higher recall value because the bank would be more interested in knowing the actual positives i.e. number of customers who would likely be a defaulter for the next month.
The F1 score (also F-score or F-measure) is a measure of a test's accuracy. It considers both the precision p and the recall r of the test to compute the score. There are two other measures namely F2, which weighs recall higher than precision (by placing more emphasis on false negatives), and F0.5, which weighs recall lower than precision (by attenuating the influence of false negatives).
Thus, the beta value would be 2.

Weighted F- Measure
Models	    Weighted F-measure
C5.0	        0.366698825
C5.0Cost	0.355532394
Rpart    	0.341257727
Random Forest	0.392959322

Recommendation: As per the weighted F-measure, Random Forest is the best classifier.  This would help the bank to identify its potential risk customers, monitor proactively and take actions as required.



                                      



