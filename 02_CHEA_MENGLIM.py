import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from question import questions
from rich.align import Align
from rich.console import Console
from rich.progress import Progress
from rich.panel import Panel
import time  # For simulating the loading animation

# Set up the rich console for styled output
console = Console()

pd.set_option('future.no_silent_downcasting', True)

file_path = r'D:\Korea Software HRD Center\Decision Tree Homework\major_observation.csv'

# Load the dataset into a pandas DataFrame
df = pd.read_csv(file_path)

# Features: Drop the 'Observation' and 'Major' columns, and replace 'yes' with 1 and 'no' with 0
feature = df.drop(columns=['Observation', 'Major']).replace({'yes': 1, 'no': 0})

# Target: Use 'Major' as the target variable
target = df['Major']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, random_state=42)

# Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier()

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Question to ask students
questions = questions

# Feature list that corresponds to the custom questions
feature_list = [
    'Analytical', 'Creative', 'Scientific', 'Teamwork', 'Numbers', 'Logical', 
    'Helping', 'WorkingAlone', 'Writing', 'Leadership', 'PublicSpeaking', 
    'Languages', 'HandsOn', 'DetailOriented', 'ExploringIdeas', 'Structured', 
    'Technology', 'Outdoor', 'Pressure', 'Debate'
]

# Function to collect user responses
def collect_user_responses():
    responses = {}
    
    console.print("\n[bold yellow]Please answer the following questions with 'yes' or 'no':[/bold yellow]\n")
    
    for question, feature in zip(questions, feature_list):
        response = console.input(f"[bold cyan]{question}[/bold cyan] (yes/no): ").lower().strip()
        while response not in ['yes', 'no']:
            response = console.input(f"[bold red]Invalid response.[/bold red] Please answer 'yes' or 'no'. {question} (yes/no): ").lower().strip()
        # Map 'yes' to 1 and 'no' to 0
        responses[feature] = 1 if response == 'yes' else 0
    
    return responses

# Function to predict the user's major based on their responses
def predict_major(user_responses):
    # Convert the user responses to a DataFrame to match the model's input format
    user_data = pd.DataFrame([user_responses])
    
    # Progress bar for model prediction
    with Progress() as progress:
        task = progress.add_task("[green]Predicting major...", total=100)
        while not progress.finished:
            progress.update(task, advance=20)
            time.sleep(0.1)  # Simulating time for the prediction process
    
    # Predict the major using the trained model
    predicted_major = clf.predict(user_data)
    
    return predicted_major

# Function to display the main menu
def main_menu():
    while True:
        console.clear()
        console.print("[yellow]1. Get Major Recommendation[/yellow]")
        console.print("[red]2. Exit[/red]")
        
        choice = console.input("[bold cyan]Please choose an option (1 or 2): ").strip()
        
        if choice == '1':
            # Clear the console and display the welcome message
            console.clear()
            welcome_text = "[bold magenta]Welcome to Major Prediction System[/bold magenta]"
            welcome_panel = Panel(
                welcome_text,
                border_style="green",
                style="yellow",
                padding=(1, 5),
                expand=False
            )

            console.print(Align.center(welcome_panel))
            console.print("\n[bold yellow]Press any key to start...[/bold yellow]")
            console.input()

            # Simulate a loading process
            with Progress() as progress:
                task = progress.add_task("[yellow]Loading, please wait...[/yellow]", total=100)
                while not progress.finished:
                    progress.update(task, advance=10)
                    time.sleep(0.1)  # Simulate loading time

            # Move on to the questions
            user_responses = collect_user_responses()
            predicted_major = predict_major(user_responses)

            # Display the prediction result
            console.rule("[bold green]Prediction Complete[/bold green]")
            console.print(f"[bold yellow]\nBased on your responses, your predicted major is:[/bold yellow] [bold magenta]{predicted_major[0]}[/bold magenta]")
            console.print("\n[bold yellow]Press any key to return to the menu...[/bold yellow]")
            console.input()
        
        elif choice == '2':
            # Exit message
            console.clear()
            exit_message = "[bold red]Thank you for using the Major Prediction System! Goodbye![/bold red]"
            console.print(Panel(exit_message, border_style="green"))
            break
        
        else:
            console.print("[bold red]Invalid option. Please choose 1 or 2.[/bold red]")
            time.sleep(1)

# Start the main menu
main_menu()
