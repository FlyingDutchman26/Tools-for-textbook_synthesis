import random

audience = ['programming beginners',
            'intermediate programming learners',
            'Beginner Programmers',
            'Python Programmers',
            'Python learners',
            'Advanced Programmers',
            'Experienced programmers',
            'professional programmers',
            'students majoring in deep learning',
            'students majoring in machine learning',
            'students majoring in mathematics',
            'undergraduate students majoring in computer science',
            'teenagers interested in computers',
            'graduate students in computer science',
            'Specialized Programmers',
            'amateur computer enthusiast',
            'Non-computer science major students',
            'Self-taught computer learners',
            "Deep Learning Engineer",
            "Machine Learning Engineer",
            "Data Scientist",
            "Software Engineer",
            "Full-Stack Developer",
            "Web Developer",
            "Mobile App Developer",
            "Cloud Engineer",
            "DevOps Engineer",
            "Algorithm Developer",
            "Blockchain Developer",
            "Teenage Programming Enthusiasts",
            "Computer Enthusiasts",
            "Mathematics Students",
            "Non-Computer Science STEM Students",
            "Financial Quantitative Analysts",
            "Data Science Aspirants",
            "Web Development Enthusiasts",
            "Natural Language Processing Enthusiasts",
            "Machine Learning Enthusiasts",
            "AI Researchers",
            "NLP Beginners",
            "Scientific Researchers Using Python",
            "Educators Teaching Coding",
            "Python enthusiast"]


basic_concepts_fundamentals = [
    "Using Python interpreter",
    "Invoking the Interpreter (Argument Passing,Interactive Mode)",
    "Source Code Encoding"
    "Code Comments",
    "Variables",
    "Data Types (int, float, str, bool)",
    "Type Conversion",
    "Operators (+, -, *, /, //, %, **)",
    "Print Statement",
    "Input Function",
    "Conditional Statements (if, elif, else)",
    "Loops (for, while, break, continue)",
    "The del statement",
    "pass and match Statements",
    "Lists, Tuples, Sets, Dictionaries",
    "List Comprehensions",
    "Defining Functions",
    "Scope and Global/Local Variables",
    "Defining Functions(Default Argument Values,Keyword Arguments,Arbitrary Argument Lists,Unpacking Argument Lists,Documentation Strings)",
    "Defining Functions:Special parameters:(Positional-or-Keyword Arguments,Positional-Only Parameters,Keyword-Only Arguments,Function Examples,Recap)",
    "Function Annotations"
    "Lambda Functions",
    "Modules and Packages",
    "Importing Modules",
    "Namespaces",
    
]

string_regular_expression = [
    "String Concatenation",
    "String Indexing and Slicing",
    "String Methods (split, join, replace, find, etc.)",
    "Escape Characters",
    "Formatting Strings",
    "Regular Expressions (re module)"
]

advanced_syntax = [
    "Generators",
    "Iterators",
    "Decorators",
    "Context Managers (with statement)",
    "Unpacking and Packing",
    "Yield Statement",
    "Walrus Operator (:=)"
]

file_handling = [
    "Opening and Closing Files",
    "Reading and Writing Text Files",
    "Binary File I/O",
    "CSV and JSON Files",
    "Exception Handling in File Operations",
    "Syntax Errors",
]

common_packages_libraries = [
    "NumPy",
    "pandas",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "TensorFlow",
    "PyTorch",
    "NLTK",
    "requests",
    "BeautifulSoup"
]

error_handling_debugging = [
    "Try-Except Blocks",
    "Handling Multiple Exceptions",
    "Raising Exceptions",
    "Debugging Techniques",
    "Logging"
]

algorithms_data_structures = [
    "Sorting Algorithms (BubbleSort, MergeSort, QuickSort, etc.)",
    "Searching Algorithms (Binary Search, Linear Search)",
    "Stacks, Queues, Linked Lists, Trees",
    "Graphs and Graph Algorithms",
    "Hashing and Hashmaps"
]

object_oriented_programming = [
    "Classes and Objects",
    "Constructors and Destructors",
    "Inheritance and Polymorphism",
    "Encapsulation",
    "Method Overriding"
]

gui_programming = [
    "Tkinter",
    "PyQt",
    "GUI Widgets",
    "Event Handling"
]

networking_web_programming = [
    "Socket Programming",
    "HTTP Requests (requests library)",
    "Web Frameworks (Flask)",
    "Web Frameworks (Django)",
    "Web Frameworks (Fastapi)"
    "RESTful APIs"
]

mobile_app_development = [
    "Kivy (Cross-platform)",
    "Java Bridge (Jython for Android)"
]

system_programming = [
    "OS Module",
    "Subprocess Module",
    "Environment Variables",
    "Working with Files and Directories"
]

concurrency_parallelism = [
    "Threading",
    "Multiprocessing"
]

database_programming = [
    "SQLite",
    "MySQL",
    "PostgreSQL",
    "CRUD Operations"
]

multimedia_programming = [
    "Pillow (Image Processing)",
    "Pygame (Game Development)"
]

reverse_engineering_analysis = [
    "Decompilation",
    "Disassembly",
    "Debugging Tools"
]

data_analysis_visualization = [
    "Data Cleaning and Transformation",
    "Exploratory Data Analysis (EDA)",
    "Statistical Analysis",
    "Data Visualization"
]

cryptography_programming = [
    "Hashing Algorithms",
    "Encryption and Decryption",
    "Public Key Infrastructure (PKI)"
]

numerical_analysis_linear_algebra = [
    "scipy",
    "Linear Equations",
    "Eigenvalues and Eigenvectors",
]

machine_deep_learning = [
    "how to use pytorch",
    "MLP",
    "Classification and Regression",
    "Neural Networks",
    "Convolutional Neural Networks (CNN)",
    "Recurrent Neural Networks (RNN)",
    "Transfer Learning",
    "huggingface",
    "how to train a neutral network and use it to inference",
    "encoder-decoder-model",
    'Seq2Seq model',
    'ResNet',
    'LSTM',
    'GRU',
    'attention-mechanism',
]

computer_vision = [
    "Image Processing",
    "Feature Extraction",
    "Object Detection",
    "Image Segmentation",
    "Image Caption",
    "Visual Transformer"
]

natural_language_processing = [
    "Tokenization",
    "Part-of-Speech Tagging",
    "Named Entity Recognition",
    "Sentiment Analysis",
    "Visual Transformer",
    "self-attention mechanism",
    "transformer model",
    "machine translation with neural network",
    "construct a language model",
    
]

statistical_learning = [
    "Regression Analysis",
    "Clustering",
    "Dimensionality Reduction"
]

# python doc
using_python_interpreter = [
    "Invoking the Interpreter",
    "Argument Passing",
    "Interactive Mode",
    "The Interpreter and Its Environment",
    "Source Code Encoding"
]

informal_introduction_python = [
    "Using Python as a Calculator",
    "Numbers",
    "Text",
    "Lists",
    "First Steps Towards Programming"
]

control_flow_tools = [
    "if Statements",
    "for Statements",
    "The range() Function",
    "break and continue Statements, and else Clauses on Loops",
    "pass Statements",
    "match Statements",
    "Defining Functions",
    "More on Defining Functions",
    "Default Argument Values",
    "Keyword Arguments",
    "Special parameters",
    "Positional-or-Keyword Arguments",
    "Positional-Only Parameters",
    "Keyword-Only Arguments",
    "Function Examples",
    "Recap",
    "Arbitrary Argument Lists",
    "Unpacking Argument Lists",
    "Lambda Expressions",
    "Documentation Strings",
    "Function Annotations",
    "Intermezzo: Coding Style"
]

data_structures = [
    "More on Lists",
    "Using Lists as Stacks",
    "Using Lists as Queues",
    "List Comprehensions",
    "Nested List Comprehensions",
    "The del statement",
    "Tuples and Sequences",
    "Sets",
    "Dictionaries",
    "Looping Techniques",
    "More on Conditions",
    "Comparing Sequences and Other Types"
]

modules = [
    "More on Modules",
    "Executing modules as scripts",
    "The Module Search Path",
    "“Compiled” Python files",
    "Standard Modules",
    "The dir() Function",
    "Packages",
    "Importing * From a Package",
    "Intra-package References",
    "Packages in Multiple Directories"
]

input_output = [
    "Fancier Output Formatting",
    "Formatted String Literals",
    "The String format() Method",
    "Manual String Formatting",
    "Old string formatting",
    "Reading and Writing Files",
    "Methods of File Objects",
    "Saving structured data with json"
]

errors_exceptions = [
    "Syntax Errors",
    "Exceptions",
    "Handling Exceptions",
    "Raising Exceptions",
    "Exception Chaining",
    "User-defined Exceptions",
    "Defining Clean-up Actions",
    "Predefined Clean-up Actions",
    "Raising and Handling Multiple Unrelated Exceptions",
    "Enriching Exceptions with Notes"
]

classes = [
    "A Word About Names and Objects",
    "Python Scopes and Namespaces",
    "Scopes and Namespaces Example",
    "A First Look at Classes",
    "Class Definition Syntax",
    "Class Objects",
    "Instance Objects",
    "Method Objects",
    "Class and Instance Variables",
    "Random Remarks",
    "Inheritance",
    "Multiple Inheritance",
    "Private Variables",
    "Odds and Ends",
    "Iterators",
    "Generators",
    "Generator Expressions"
]

standard_library_tour = [
    "Brief Tour of the Standard Library",
    "Operating System Interface",
    "File Wildcards",
    "Command Line Arguments",
    "Error Output Redirection and Program Termination",
    "String Pattern Matching",
    "Mathematics",
    "Internet Access",
    "Dates and Times",
    "Data Compression",
    "Performance Measurement",
    "Quality Control",
    "Batteries Included",
    "Output Formatting",
    "Templating",
    "Working with Binary Data Record Layouts",
    "Multi-threading",
    "Logging",
    "Weak References",
    "Tools for Working with Lists",
    "Decimal Floating Point Arithmetic"
]

virtual_environments_packages = [
    "Virtual Environments and Packages",
    "Introduction",
    "Creating Virtual Environments",
    "Managing Packages with pip"
]


# others

common_data_structures = [
    "Arrays",
    "Linked Lists",
    "Doubly Linked Lists",
    "Circular Linked Lists",
    "Stacks (LIFO - Last In First Out)",
    "Queues (FIFO - First In First Out)",
    "Priority Queues",
    "Hash Tables",
    "Binary Trees",
    "Binary Search Trees",
    "AVL Trees",
    "Red-Black Trees",
    "B-Trees",
    "Heap Trees",
    "Trie (Prefix Tree)",
    "Suffix Array",
    "Suffix Tree",
    "Directed Graphs",
    "Undirected Graphs",
    "Weighted Graphs",
    "Bipartite Graphs",
    "Disjoint-Set (Union-Find)",
    "Bloom Filters",
    "Skip Lists",
    "Sets",
    "Maps",
    "Matrices",
    "Strings"
]

common_algorithms = [
    "Linear Search",
    "Binary Search",
    "Hashing",
    "Bubble Sort",
    "Selection Sort",
    "Insertion Sort",
    "Merge Sort",
    "Quick Sort",
    "Heap Sort",
    "Radix Sort",
    "Counting Sort",
    "Factorial (Recursion)",
    "Fibonacci Sequence (Recursion)",
    "Tower of Hanoi (Recursion)",
    "Longest Common Subsequence (Dynamic Programming)",
    "Knapsack Problem (Dynamic Programming)",
    "Coin Change Problem (Greedy)",
    "Fractional Knapsack Problem (Greedy)",
    "Huffman Coding (Greedy)",
    "Binary Search (Divide and Conquer)",
    "Closest Pair of Points (Divide and Conquer)",
    "Breadth-First Search (BFS) (Graph)",
    "Depth-First Search (DFS) (Graph)",
    "Dijkstra's Algorithm (Graph)",
    "Bellman-Ford Algorithm (Graph)",
    "Floyd-Warshall Algorithm (Graph)",
    "Topological Sorting (Graph)",
    "Minimum Spanning Tree (Prim's and Kruskal's) (Graph)",
    "Naive String Searching",
    "Rabin-Karp Algorithm (String)",
    "KMP Algorithm (String)",
    "Boyer-Moore Algorithm (String)"
]


linear_algebra = [
    "Vector representation and operations",
    "Matrix representation and operations",
    "Scalar multiplication",
    "Vector addition and subtraction",
    "Matrix addition and subtraction",
    "Matrix-vector multiplication",
    "Matrix-matrix multiplication",
    "Transpose of a matrix",
    "Determinant of a matrix",
    "Inverse of a matrix",
    "Eigenvalues and eigenvectors",
    "Dot product",
    "Cross product",
    "Orthogonalization and Gram-Schmidt process",
    "Vector spaces",
    "Basis and dimension",
    "Linear transformations",
    "Kernel (null space) and image (column space)",
    "Rank and nullity of a matrix",
    "LU decomposition",
    "QR decomposition",
    "Singular value decomposition (SVD)",
    "Eigenvalue decomposition",
    "Projection matrices",
    "Least squares approximation",
    "Matrix factorizations",
    "Inner product spaces",
    "Orthogonal and orthonormal vectors",
    "Hermitian (conjugate transpose) matrices",
    "Positive definite matrices",
    "Diagonalization of matrices",
    "Change of basis",
    "Jordan canonical form",
    "Matrix norms",
    "Matrix equations and linear systems",
    "Cramer's rule",
    "Norms of vectors",
    "Vector spaces and subspaces"
]


machine_learning_concepts = [
    "Supervised learning",
    "Unsupervised learning",
    "Semi-supervised learning",
    "Reinforcement learning",
    "Classification",
    "Regression",
    "Clustering",
    "Dimensionality reduction",
    "Feature engineering",
    "Cross-validation",
    "Bias-Variance tradeoff",
    "Overfitting and underfitting",
    "Hyperparameter tuning",
    "Ensemble methods",
    "Decision trees",
    "Random forests",
    "Gradient boosting",
    "Support Vector Machines (SVM)",
    "Neural networks",
    "Deep learning",
    "Convolutional Neural Networks (CNN)",
    "Recurrent Neural Networks (RNN)",
    "Natural Language Processing (NLP)",
    "Transfer learning",
    "Regularization",
    "Loss functions",
    "Activation functions",
    "Optimization algorithms (e.g., SGD, Adam)",
    "Batch normalization",
    "Data preprocessing",
    "Data augmentation",
    "Evaluation metrics (e.g., accuracy, precision, recall, F1-score)",
    "Confusion matrix",
    "Bias and fairness in machine learning",
    "Anomaly detection",
    "Time series analysis",
    "Recommendation systems",
    "Principal Component Analysis (PCA)",
    "K-Means clustering",
    "Nearest Neighbor methods",
    "Autoencoders",
    "Generative Adversarial Networks (GANs)",
    "Reinforcement learning algorithms (e.g., Q-learning, DDPG)",
    "Markov models",
    "Gradient Descent and variants",
    "Kernel methods",
    "Word embeddings",
    "Neural architecture design",
    "Data imbalance handling",
    "Neuroevolution",
    "Model interpretation and explainability"
]

simple_probability_statistics_concepts = [
    "Probability calculations",
    "Dice roll simulation",
    "Coin flip simulation",
    "Mean, median, and mode calculations",
    "Histogram plotting",
    "Normal distribution visualization",
    "Random variable generation",
    "Simple hypothesis testing",
    "Basic correlation analysis",
    "Simple linear regression",
    "Sampling from a population",
    "Basic descriptive statistics",
    "Calculating standard deviation",
    "Expected value calculation",
    "Bernoulli trials simulation",
    "Binomial distribution calculations",
    "Poisson distribution calculations",
    "Basic Bayes' theorem application",
    "Basic confidence interval calculation",
    "Basic chi-squared test",
    "Basic ANOVA analysis",
    "Basic probability trees",
    "Simple Monte Carlo simulation",
    "Simple random sampling",
    "Basic probability density function plot",
    "Simple resampling (e.g., bootstrapping)"
]

essential_database_programming_concepts = [
    "Connecting to a database",
    "Creating tables and defining schemas",
    "Inserting data into a table",
    "Querying data with SQL",
    "Updating and deleting records",
    "Basic data validation and sanitization",
    "Using parameterized queries",
    "Working with relational databases",
    "Database transactions",
    "Basic error handling in database operations",
    "Database authentication and security",
    "Using an ORM (Object-Relational Mapping)",
    "Executing raw SQL queries",
    "Fetching and displaying query results",
    "Exporting and importing data",
    "Database connection pooling",
    "Simple data visualization from database results"
]

essential_web_dev_and_networking = [
    "Creating a basic web server with Flask",
    "Handling HTTP requests and responses",
    "Using RESTful API principles",
    "Fetching data from external APIs",
    "Implementing user authentication and sessions",
    "Interacting with databases using Python",
    "Working with HTML, CSS, and JavaScript",
    "Using web frameworks like Django or Flask",
    "Deploying a web app with cloud platforms",
    "Understanding URL routing and patterns",
    "Basic front-end and back-end integration",
    "Building a basic chat application",
    "Implementing basic security measures",
    "Understanding HTTP methods and status codes",
    "Using sockets for network communication",
    "Creating a simple client-server application",
    "Implementing network protocols with Python",
    "Sending and receiving data over sockets",
    "Basic error handling in network programming",
    "Understanding TCP/IP and UDP",
    "Creating a basic web scraper",
    "Working with network APIs and libraries",
    "Handling DNS resolution and requests",
    "Implementing network monitoring tools",
    "Capturing and analyzing network traffic"
]

essential_network_security_and_cryptography_with_python = [
    "Implementing basic encryption and decryption",
    "Hashing passwords for secure storage",
    "Creating digital signatures with Python",
    "Working with secure communication protocols",
    "Basic network vulnerability assessment",
    "Implementing secure user authentication",
    "Understanding and preventing DDoS attacks",
    "Using Python for penetration testing",
    "Securing sensitive data with encryption",
    "Implementing two-factor authentication",
    "Basic intrusion detection techniques",
    "Creating secure login mechanisms",
    "Understanding SSL/TLS and certificates",
    "Implementing secure file transfers",
    "Exploring cybersecurity threats"
]

simple_math_concepts_with_python = [
    "Calculating the area of geometric shapes",
    "Finding the square root of a number",
    "Generating multiplication tables",
    "Calculating the factorial of a number",
    "Checking if a number is prime",
    "Calculating the greatest common divisor (GCD)",
    "Solving basic linear equations",
    "Converting units (e.g., inches to centimeters)",
    "Calculating percentages and ratios",
    "Generating random numbers within a range",
    "Working with basic geometric transformations",
    "Calculating the perimeter of shapes",
    "Using the Pythagorean theorem",
    "Solving basic quadratic equations",
    "Converting between different number bases",
    "Calculating simple interest and compound interest",
    "Calculating the circumference of a circle",
    "Solving basic geometry problems",
    "Finding the LCM of two or more numbers",
    "Performing basic arithmetic operations",
    "Calculating the volume of simple shapes",
    "Working with angles and degrees",
    "Using basic trigonometric functions",
    "Calculating the mean, median, and mode"
]

essential_image_processing_data_analysis_with_python = [
    "Loading and displaying images with Python",
    "Applying basic image filters and transformations",
    "Creating histograms and analyzing pixel intensity",
    "Image cropping, resizing, and rotation",
    "Implementing edge detection algorithms",
    "Working with color spaces and channels",
    "Creating image mosaics and collages",
    "Basic image segmentation techniques",
    "Exploring data with pandas DataFrames",
    "Calculating descriptive statistics",
    "Visualizing data with matplotlib",
    "Creating line, bar, and scatter plots",
    "Generating pie and donut charts",
    "Creating heatmaps and contour plots",
    "Plotting time series data",
    "Performing basic statistical analysis",
    "Implementing data filtering and manipulation",
    "Working with dates and times in data",
    "Using NumPy for numerical computations",
    "Performing matrix operations and linear algebra",
    "Working with large datasets efficiently",
    "Creating interactive data visualizations with Plotly",
    "Implementing basic data clustering",
    "Working with geospatial data and maps",
    "Generating animated visualizations"
]

essential_object_oriented_programming_with_python = [
    "Creating and using classes",
    "Defining attributes and methods",
    "Understanding encapsulation and abstraction",
    "Implementing inheritance and polymorphism",
    "Creating class constructors and destructors",
    "Working with class instances and objects",
    "Using class properties and methods",
    "Understanding class access modifiers",
    "Implementing class composition",
    "Creating and using class decorators",
    "Using magic methods (dunder methods)",
    "Working with static and class methods",
    "Implementing interfaces and abstract classes",
    "Understanding the 'self' keyword in Python",
    "Using super() for method overriding",
    "Exploring multiple inheritance",
    "Creating design patterns with classes",
    "Using composition over inheritance",
    "Implementing simple inheritance hierarchies",
    "Understanding method overloading and overriding",
    "Working with class attributes and instance attributes",
    "Creating and using class instances",
    "Using inheritance to extend functionality",
    "Implementing factory patterns with classes"
]


beginner_friendly_python_practice = [
    "Printing messages to the console",
    "Simple arithmetic calculations",
    "Working with variables and data types",
    "Using conditional statements (if, elif, else)",
    "Creating loops (for and while)",
    "Basic string manipulation",
    "Creating and using functions",
    "Working with lists and arrays",
    "Understanding and using dictionaries",
    "Creating and using classes",
    "Using libraries and modules",
    "Handling user input",
    "Creating basic text-based games",
    "Simple file I/O operations",
    "Implementing basic error handling",
    "Exploring Python's standard library",
    "Creating simple web scraping scripts",
    "Generating random numbers and values",
    "Creating basic data visualizations",
    "Working with external APIs",
    "Creating a basic calculator",
    "Writing and running test cases",
    "Understanding scope and namespaces",
    "Implementing basic algorithms (e.g., sorting)",
    "Creating a simple to-do list application"
]


beginner_practical_python_practice = [
    "Printing patterns using loops",
    "Calculating the area of geometric shapes",
    "Implementing a basic calculator",
    "Converting units (e.g., Celsius to Fahrenheit)",
    "Creating a simple guess the number game",
    "Calculating the factorial of a number",
    "Building a basic to-do list application",
    "Generating simple multiplication tables",
    "Creating a basic text-based quiz",
    "Finding the prime numbers in a range",
    "Creating a basic alarm clock",
    "Calculating the Fibonacci sequence",
    "Implementing a basic grade calculator",
    "Creating a simple text editor",
    "Building a basic currency converter",
    "Creating a simple calculator for tips",
    "Implementing a basic countdown timer",
    "Generating a basic random password",
    "Creating a basic contact book",
    "Building a simple password manager",
    "Implementing a basic dice rolling simulator",
    "Creating a basic ASCII art generator",
    "Building a basic calculator for area and perimeter",
    "Implementing a simple shopping list",
    "Creating a basic BMI calculator"
]

intermediate_python_practice = [
    "Implementing data structures (linked lists, stacks, queues)",
    "Creating a basic text-based RPG game",
    "Building a simple blog application",
    "Creating a web scraper for data collection",
    "Implementing a basic chatbot",
    "Developing a simple RESTful API",
    "Creating a command-line tool",
    "Building a basic social media platform",
    "Implementing data visualization dashboards",
    "Creating a simple e-commerce website",
    "Building a personal finance tracker",
    "Implementing machine learning algorithms (e.g., regression)",
    "Developing a weather forecasting application",
    "Creating a basic data analysis tool",
    "Building a recommendation system",
    "Implementing a basic neural network",
    "Developing a portfolio website",
    "Creating a simple content management system (CMS)",
    "Building a simple task management application",
    "Implementing a basic image recognition system",
    "Developing a personal blog platform",
    "Creating a basic music recommendation app",
    "Building a finance management dashboard",
    "Implementing a sentiment analysis tool",
    "Developing a basic booking system"
]


intermediate_pytorch_practice = [
    "Implementing custom neural network architectures",
    "Creating a deep learning model for image classification",
    "Building a convolutional neural network (CNN)",
    "Developing a recurrent neural network (RNN) for sequence data",
    "Creating a generative adversarial network (GAN)",
    "Implementing transfer learning for fine-tuning models",
    "Building a natural language processing (NLP) model",
    "Developing a style transfer application",
    "Creating a reinforcement learning agent",
    "Implementing a variational autoencoder (VAE)",
    "Building a text generation model using LSTM or GRU",
    "Developing a sentiment analysis model",
    "Creating a recommendation system with embeddings",
    "Implementing neural style transfer",
    "Building a language translation model",
    "Developing a stock price prediction model",
    "Creating a character-level language model",
    "Implementing object detection using YOLO or SSD",
    "Building a speech recognition system",
    "Developing a time series forecasting model",
    "Creating a facial recognition application",
    "Implementing a multi-modal fusion model",
    "Building a medical image analysis tool",
    "Developing a question answering system"
]

numpy_practice_for_beginners = [
    "Creating and manipulating arrays",
    "Performing basic arithmetic operations",
    "Using array slicing and indexing",
    "Calculating basic statistics",
    "Reshaping and flattening arrays",
    "Implementing broadcasting operations",
    "Applying element-wise functions",
    "Using boolean indexing and masking",
    "Working with multi-dimensional arrays",
    "Generating arrays of random numbers",
    "Performing matrix multiplication",
    "Implementing aggregation functions",
    "Understanding data types and dtype",
    "Creating arrays from existing data",
    "Using universal functions (ufuncs)",
    "Applying logical and comparison operations",
    "Working with missing values (NaN)",
    "Using vectorized operations for efficiency",
    "Implementing basic linear algebra",
    "Creating arrays with specified ranges",
    "Understanding array views and copies",
    "Creating structured arrays"
]

intermediate_python_practice_for_advancement = [
    "Building a simple web scraper to collect data",
    "Creating a command-line tool for file manipulation",
    "Developing a basic text-based game or quiz",
    "Implementing a basic recommendation system",
    "Building a currency converter with real-time exchange rates",
    "Creating a weather forecasting application using APIs",
    "Developing a basic personal finance tracker",
    "Implementing a simple sentiment analysis tool",
    "Building a basic portfolio website",
    "Creating a data visualization dashboard",
    "Developing a simple task management application",
    "Implementing a password generator with user-defined criteria",
    "Building a simple content management system (CMS) for blogs",
    "Creating a basic chatbot using natural language processing",
    "Developing a basic e-commerce website with user authentication",
    "Implementing a basic image recognition application",
    "Building a personal blog platform with user profiles",
    "Creating a basic social media platform for sharing content",
    "Developing a basic music recommendation app",
    "Implementing a time series forecasting tool",
    "Building a simple API to retrieve and display data",
    "Creating a basic movie or book recommendation system",
    "Developing a personal finance management dashboard",
    "Implementing a simple machine learning model",
    "Building a recipe-sharing platform"
]


categories = {
    #  gpt-3.5 - origin
    "basic_concepts_fundamentals": basic_concepts_fundamentals,
    "string_regular_expression": string_regular_expression,
    "advanced_syntax": advanced_syntax,
    "file_handling": file_handling,
    "common_packages_libraries": common_packages_libraries,
    "error_handling_debugging": error_handling_debugging,
    "algorithms_data_structures": algorithms_data_structures,
    "object_oriented_programming": object_oriented_programming,
    "gui_programming": gui_programming,
    "networking_web_programming": networking_web_programming,
    "mobile_app_development": mobile_app_development,
    "system_programming": system_programming,
    "concurrency_parallelism": concurrency_parallelism,
    "database_programming": database_programming,
    "multimedia_programming": multimedia_programming,
    "reverse_engineering_analysis": reverse_engineering_analysis,
    "data_analysis_visualization": data_analysis_visualization,
    "cryptography_programming": cryptography_programming,
    "numerical_analysis_linear_algebra": numerical_analysis_linear_algebra,
    "machine_deep_learning": machine_deep_learning,
    "computer_vision": computer_vision,
    "natural_language_processing": natural_language_processing,
    "statistical_learning": statistical_learning,
    # python docs
    "using_python_interpreter": using_python_interpreter,
    "informal_introduction_python": informal_introduction_python,
    "control_flow_tools": control_flow_tools,
    "data_structures": data_structures,
    "modules": modules,
    "input_output": input_output,
    "errors_exceptions": errors_exceptions,
    "classes": classes,
    "standard_library_tour": standard_library_tour,
    "virtual_environments_packages": virtual_environments_packages,
    # gpt3.5- specially- prompt
    "common_data_structures":common_data_structures,
    "common_algorithms":common_algorithms,
    "linear_algebra":linear_algebra,
    "machine_learning_concepts":machine_learning_concepts,
    "simple_probability_statistics_concepts":simple_probability_statistics_concepts,
    "essential_database_programming_concepts":essential_database_programming_concepts,
    "essential_web_dev_and_networking":essential_web_dev_and_networking,
    "essential_network_security_and_cryptography_with_python":essential_network_security_and_cryptography_with_python,
    "simple_math_concepts_with_python":simple_math_concepts_with_python,
    "essential_image_processing_data_analysis_with_python":essential_image_processing_data_analysis_with_python,
    "essential_object_oriented_programming_with_python":essential_object_oriented_programming_with_python,
    "beginner_friendly_python_practice":beginner_friendly_python_practice,
    "beginner_practical_python_practice":beginner_practical_python_practice,
    "intermediate_python_practice":intermediate_python_practice,
    "intermediate_pytorch_practice":intermediate_pytorch_practice,
    "numpy_practice_for_beginners":numpy_practice_for_beginners,
    "intermediate_python_practice_for_advancement":intermediate_python_practice_for_advancement,
}


def generate_topics():
    topics = []
    for key,value in categories.items():
        for item in value:
            topics.append(key+': '+item)
    return topics


def synthesize_textbook(topics, num_topics = 6):
    selected_topics = random.sample(topics,num_topics)
    random.shuffle(selected_topics)
    topic_str = ''
    for topic in selected_topics:
        topic_str += topic + '\n'
        
    prompt = f"You are an experienced Python programmer who has taken on the role of a patient instructor, \
tasked with crafting comprehensive and informative python textbooks for {random.choice(audience)}. \
Your current focus is on creating python tutorials that feature code examples with annotations. \
The aim is to facilitate optimal learning for students by addressing multiple topics with rich code example, \
thereby enabling them to acquire a broader scope of knowledge. \
Your task involves selecting at least two subjects from the provided list below \
and creating a coherent code script accompanied by textbook-quality explanations.\
The topics available for selection are as follows:\n\
{topic_str}\
The choice and order of topics is at your discretion, so long as you deem them compatible for integration into a cohesive tutorial. \
Also, the topics you choose should be suitable and helpful for {random.choice(audience)} who are reading your textbooks. \
You are also welcome to introduce supplementary knowledge that enhances the tutorial's content. \
Remember, providing high-quality inline code comments is strongly recommended! \
Please avoid excessive reliance on external libraries or packages; instead, focus on utilizing code and comments to implement the inherent logic.\
Should you employ functions from any packages necessary, be sure to furnish explanations for their usage. \
Please focus more on generating abundant code examples. Summaries or conclusions are not necessary. \
You don't need to respond or comment to this instruction. Just generate the content of your tutorial."
    return prompt
    
if __name__ == '__main__':
    topics = generate_topics()
    print('total_topic_numbers: ',len(topics))
    prompt = synthesize_textbook(topics, num_topics=6)
    print(prompt)
    
    

