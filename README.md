# handwriting_recognizer

Installations: Only Matlab tool needed ( no library methods used).
No other packages used.


Steps to run and test the given test samples:

1. Download the 'project.zip' folder and extract the 'project' folder as a folder or unzip the contents of the folder in one folder to your Matlab 'userpath' directory.
2. Paste the test data file say 'TEST.data' for example, to be tested, in this folder where you've unzipped the 'project' folder.
3. Open the file 'project.m' in Matlab 
4. Copy the name and extension of the sample 'TEST.data' file to be tested and paste it in the line number- '5' and '17' of 'project.m'.
5. Then start running this 'project.m' file in Matlab.
6. In the command prompt of Matlab, it will ask you to enter '1' or '2' like this: 'Enter: 1 (for Testing) or 2 (for Traning and Testing)'.
7. Type in '1'-if you just want to test the 'TEST.data' data file and hit enter . 
8. Else, for both training and then testing by the improved model, press '2' from command prompt and Enter.
9. The output of the classlabels 'classlabels.txt' will be produced in that same folder where you've unzipped the 'project' folder.
10. Open the 'classlabels.txt' file to see the class labels.




Example:
Note: I assumed 'TEST.data' name for final test data to check my model.

1> Pasting 'TEST.data' in 'project' folder.
2> In 'project.m', copy and pasting the name- 'TEST.data' at line numbers:  5 and 17                     
3> Run --> MATLAB>Enter: 1 (for Testing) or 2 (for Traning and Testing) 1
4> Checking 'classlabels.txt' for labels.
