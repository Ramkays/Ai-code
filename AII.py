
#N queens

global N
N = 4


def printSolution(board):
	for i in range(N):
		for j in range(N):
			if board[i][j] == 1:
				print("Q",end=" ")
			else:
				print(".",end=" ")
		print()


def isSafe(board, row, col):

	for i in range(col):
		if board[row][i] == 1:
			return False

	for i, j in zip(range(row, -1, -1),
					range(col, -1, -1)):
		if board[i][j] == 1:
			return False

	for i, j in zip(range(row, N, 1),
					range(col, -1, -1)):
		if board[i][j] == 1:
			return False

	return True


def solveNQUtil(board, col):

	if col >= N:
		return True
	for i in range(N):

		if isSafe(board, i, col):

			board[i][col] = 1

			if solveNQUtil(board, col + 1) == True:
				return True
			board[i][col] = 0

	return False


def solveNQ():
	board = [[0, 0, 0, 0],
			[0, 0, 0, 0],
			[0, 0, 0, 0],
			[0, 0, 0, 0]]

	if solveNQUtil(board, 0) == False:
		print("Solution does not exist")
		return False

	printSolution(board)
	return True


if __name__ == '__main__':
	solveNQ()
. . Q . 
Q . . . 
. . . Q 
. Q . . 
#Camel
#Camel
ban=int(input('Enter no. of bananas : '))
dis=int(input('Enter distance : '))
cap=int(input('Enter max capacity of your camel : '))
lose=0
start=ban
for i in range(dis):
    
    while start>0:
        start=start-cap

        if start==1:
            lose=lose-1

        lose=lose+2

    lose=lose-1
    start=ban-lose
    
    if start==0:
        break
print("No of Trips : ",start)
Enter no. of bananas : 3000
Enter distance : 1000
Enter max capacity of your camel : 1000
No of Trips :  0
#crpyt
def find_value(word, assigned):
  num = 0
  for char in word:
    num = num * 10
    num += assigned[char]
  return num
def is_valid_assignment(word1, word2, result, assigned):
  # First letter of any word cannot be zero.
  if assigned[word1[0]] == 0 or assigned[word2[0]] == 0 or assigned[result[0]] == 0:
    return False
  return True
def _solve(word1, word2, result, letters, assigned, solutions):
  if not letters:
    if is_valid_assignment(word1, word2, result, assigned):
      num1 = find_value(word1, assigned)
      num2 = find_value(word2, assigned)
      num_result = find_value(result, assigned)
      if num1 + num2 == num_result:
        solutions.append((f'{num1} + {num2} = {num_result}', assigned.copy()))
    return
  for num in range(10):
    if num not in assigned.values():
      cur_letter = letters.pop()
      assigned[cur_letter] = num
      _solve(word1, word2, result, letters, assigned, solutions)
      assigned.pop(cur_letter)
      letters.append(cur_letter)
def solve(word1, word2, result):
  letters = sorted(set(word1) | set(word2) | set(result))
  if len(result) > max(len(word1), len(word2)) + 1 or len(letters) > 10:
    print('0 Solutions!')
    return
  solutions = []
  _solve(word1, word2, result, letters, {}, solutions)
  if solutions:
    print('\nSolutions:')
    for soln in solutions:
      print(f'{soln[0]}\t{soln[1]}')
if __name__ == '__main__':
  print('CRYPTARITHMETIC PUZZLE SOLVER')
  print('WORD1 + WORD2 = RESULT')
  word1 = input('Enter WORD1: ').upper()
  word2 = input('Enter WORD2: ').upper()
  result = input('Enter RESULT: ').upper()
  if not word1.isalpha() or not word2.isalpha() or not result.isalpha():
    raise TypeError('Inputs should ony consists of alphabets.')
  solve(word1, word2, result)
CRYPTARITHMETIC PUZZLE SOLVER
WORD1 + WORD2 = RESULT
Enter WORD1: SEND
Enter WORD2: MORE
Enter RESULT: MONEY

Solutions:
9567 + 1085 = 10652	{'Y': 2, 'S': 9, 'R': 8, 'O': 0, 'N': 6, 'M': 1, 'E': 5, 'D': 7}
#graph
def isSafe(graph, color):
 
    # check for every edge
    for i in range(4):
        for j in range(i + 1, 4):
            if (graph[i][j] and color[j] == color[i]):
                return False
    return True
def graphColoring(graph, m, i, color):   
    if (i == 4):
        if (isSafe(graph, color)):           
            printSolution(color)
            return True
        return False
    for j in range(1, m + 1):
        color[i] = j       
        if (graphColoring(graph, m, i + 1, color)):
            return True
        color[i] = 0
    return False
def printSolution(color):
    print("Solution Exists:" " Following are the assigned colors ")
    for i in range(4):
        print(color[i], end=" ")
if __name__ == '__main__':
    graph = [
        [0, 1, 1, 1],
        [1, 0, 1, 0],
        [1, 1, 0, 1],
        [1, 0, 1, 0],
    ]
    m = 3  
    color = [0 for i in range(4)]   
    if (not graphColoring(graph, m, 0, color)):
        print("Solution does not exist")
Solution Exists: Following are the assigned colors 
1 2 3 2 
#BFS
def bfs(graph, start, end):
    queue = [(start, [start])]
    visited = set()

    while queue:
        (node, path) = queue.pop(0)
        if node not in visited:
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor == end:
                    return path + [neighbor]
                else:
                    queue.append((neighbor, path + [neighbor]))
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

start = 'A'
end = 'F'
print(bfs(graph, start, end))
['A', 'C', 'F']
#dfs
def dfs(graph, start, end):
    stack = [(start, [start])]
    visited = set()

    while stack:
        (node, path) = stack.pop()
        if node not in visited:
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor == end:
                    return path + [neighbor]
                else:
                    stack.append((neighbor, path + [neighbor]))
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

start = 'A'
end = 'F'
print(dfs(graph, start, end))
['A', 'C', 'F']
#best
import heapq
def best_first_search(graph, start, end, heuristic):
    heap = [(heuristic[start], start, [start])]
    visited = set()

    while heap:
        (f, node, path) = heapq.heappop(heap)
        if node not in visited:
            visited.add(node)
            if node == end:
                return path
            for neighbor in graph[node]:
                g = len(path)
                h = heuristic[neighbor]
                f = g + h
                heapq.heappush(heap, (f, neighbor, path + [neighbor]))
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

start = 'A'
end = 'E'
heuristic = {
    'A': 5,
    'B': 4,
    'C': 3,
    'D': 2,
    'E': 2,
    'F': 0
}
print(best_first_search(graph, start, end, heuristic))
['A', 'B', 'E']
#astar
import heapq
def astar(graph, start, end, heuristic):
    heap = [(heuristic[start], start, [start])]
    visited = set()

    while heap:
        (f, node, path) = heapq.heappop(heap)
        if node not in visited:
            visited.add(node)
            if node == end:
                return path
            for neighbor in graph[node]:
                g = len(path)
                h = heuristic[neighbor]
                f = g + h
                heapq.heappush(heap, (h, neighbor, path + [neighbor]))

# Example usage
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

start = 'A'
end = 'D'
heuristic = {
    'A': 5,
    'B': 4,
    'C': 3,
    'D': 2,
    'E': 2,
    'F': 0
}
print(astar(graph, start, end, heuristic))
['A', 'B', 'D']
#minmax
import math

def minimax (curDepth, nodeIndex,
      maxTurn, scores,
      targetDepth):

  if (curDepth == targetDepth):
    return scores[nodeIndex]
  
  if (maxTurn):
    return max(minimax(curDepth + 1, nodeIndex * 2,
          False, scores, targetDepth),
        minimax(curDepth + 1, nodeIndex * 2 + 1,
          False, scores, targetDepth))
  
  else:
    return min(minimax(curDepth + 1, nodeIndex * 2,
          True, scores, targetDepth),
        minimax(curDepth + 1, nodeIndex * 2 + 1,
          True, scores, targetDepth))
  

scores = [2,3,5,9,0,1,7,5]

treeDepth = math.log(len(scores), 2)

print("The optimal value is : ", end = "")
print(minimax(0, 0, False, scores, treeDepth))
The optimal value is : 5
#logic
from kanren import Relation, facts, run, var, conde, eq
parent = Relation()
facts(parent, ("Bob", "Alice"), ("Bob", "Charlie"), ("Charlie", "David"))
def grandparent(x, z):
    y = var()
    return conde((parent(x, y), parent(y, z)))
x = var()
results = run(0, x, grandparent(x, "David"))
print(results)
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
Cell In[26], line 2
      1 #logic
----> 2 from kanren import Relation, facts, run, var, conde, eq
      3 parent = Relation()
      4 facts(parent, ("Bob", "Alice"), ("Bob", "Charlie"), ("Charlie", "David"))

ModuleNotFoundError: No module named 'kanren'
#kmeans
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
wine = load_wine()
df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
scaler = StandardScaler()
X = scaler.fit_transform(df[wine.feature_names])
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
y_pred = kmeans.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title('K-Means Clustering')
plt.show()
P:\Anaconda\lib\site-packages\sklearn\cluster\_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
  warnings.warn(
P:\Anaconda\lib\site-packages\sklearn\cluster\_kmeans.py:1382: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.
  warnings.warn(

#logistic
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = load_iris()

# Create a Pandas dataframe of the features and target variable
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[iris.feature_names], df['target'], test_size=0.2, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict the classes of the test set
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy score
print("Accuracy:", accuracy)
import nltk 
import nltk.corpus
#Tokenization
from nltk.tokenize import word_tokenize 
chess = "Samay Raina is the best chess streamer in the world"  
nltk.download('punkt') 
word_tokenize(chess) #Tokenization
#sentence tokenizer 
from nltk.tokenize import sent_tokenize 
chess2 = "Samay Raina is the best chess streamer in the world. Sagar Sh ah is  the best chess coach in the world" 
sent_tokenize(chess2)
#Checking the number of tokens  
len(word_tokenize(chess))  
  #Checking the number of tokens  
len(word_tokenize(chess))
#bigrams and n-grams
astronaut = "Can anybody hear me or am I talking to myself? My mind is  running empty in the search for someone else"
astronaut_token=(word_tokenize(astronaut))#bigrams and n-grams
astronaut = "Can anybody hear me or am I talking to myself? My mind is  running empty in the search for someone else" 
astronaut_token=(word_tokenize(astronaut))
list(nltk.bigrams(astronaut_token))
list(nltk.trigrams(astronaut_token))
list(nltk.ngrams((astronaut_token),5))
#Stemming 
from nltk.stem import PorterStemmer 
my_stem = PorterStemmer()  
my_stem.stem("eating")  
my_stem.stem("going")  
my_stem.stem("shopping")
#pos-tagging 
tom ="Tom Hanks is the best actor in the world"  
tom_token = word_tokenize(tom)  
nltk.download('averaged_perceptron_tagger')  
nltk.pos_tag(tom_token)
#Named entity recognition  
from nltk import ne_chunk 
president = "Barack Obama was the 44th President of America"  
president_token = word_tokenize(president) 
president_pos = nltk.pos_tag(president_token) 
nltk.download('maxent_ne_chunker') 
nltk.download('words')  
print(ne_chunk(president_pos))
!pip install gTTS
from gtts import gTTS 
from IPython.display import Audio  
tts = gTTS('Negga')  
tts.save('1.wav') 
sound_file = '1.wav'  
Audio(sound_file, autoplay=True) 
