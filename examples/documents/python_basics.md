# Python Programming Basics

## Introduction to Python

Python is a high-level, interpreted programming language created by Guido van Rossum
and first released in 1991. It emphasizes code readability with its notable use of
significant indentation. Python is dynamically typed and garbage-collected.

## Key Features

### Easy to Learn
Python has a clean and simple syntax that makes it easy for beginners to learn
programming. The language reads almost like English, making it intuitive.

### Versatile
Python is used in many domains:
- Web development (Django, Flask)
- Data science and machine learning (NumPy, Pandas, scikit-learn)
- Automation and scripting
- Desktop applications
- Scientific computing

### Large Standard Library
Python comes with a comprehensive standard library that includes modules for:
- File I/O
- System calls
- Internet protocols
- Data compression
- Databases

## Basic Data Types

### Numbers
Python supports integers, floating-point numbers, and complex numbers.

```python
x = 42        # integer
y = 3.14      # float
z = 1 + 2j    # complex
```

### Strings
Strings in Python can be enclosed in single or double quotes.

```python
name = "Alice"
greeting = 'Hello, World!'
multiline = """This is a
multiline string"""
```

### Lists
Lists are ordered, mutable collections.

```python
fruits = ["apple", "banana", "cherry"]
fruits.append("date")
print(fruits[0])  # Output: apple
```

### Dictionaries
Dictionaries store key-value pairs.

```python
person = {
    "name": "Alice",
    "age": 30,
    "city": "New York"
}
print(person["name"])  # Output: Alice
```

## Control Flow

### If Statements
```python
x = 10
if x > 5:
    print("x is greater than 5")
elif x == 5:
    print("x equals 5")
else:
    print("x is less than 5")
```

### For Loops
```python
for i in range(5):
    print(i)

for fruit in ["apple", "banana", "cherry"]:
    print(fruit)
```

### While Loops
```python
count = 0
while count < 5:
    print(count)
    count += 1
```

## Functions

Functions are defined using the `def` keyword.

```python
def greet(name):
    """Return a greeting message."""
    return f"Hello, {name}!"

def add(a, b=0):
    """Add two numbers with a default value."""
    return a + b

# Usage
print(greet("Alice"))  # Output: Hello, Alice!
print(add(3, 4))       # Output: 7
print(add(5))          # Output: 5
```

## Classes and Objects

Python supports object-oriented programming.

```python
class Dog:
    def __init__(self, name, breed):
        self.name = name
        self.breed = breed

    def bark(self):
        return f"{self.name} says woof!"

# Create an instance
my_dog = Dog("Buddy", "Golden Retriever")
print(my_dog.bark())  # Output: Buddy says woof!
```

## File Handling

```python
# Writing to a file
with open("output.txt", "w") as f:
    f.write("Hello, World!")

# Reading from a file
with open("output.txt", "r") as f:
    content = f.read()
    print(content)
```

## Error Handling

```python
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")
finally:
    print("This always runs")
```

## Popular Python Libraries

- **NumPy**: Numerical computing with arrays
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Data visualization
- **Requests**: HTTP library for API calls
- **Django/Flask**: Web frameworks
- **TensorFlow/PyTorch**: Machine learning frameworks
