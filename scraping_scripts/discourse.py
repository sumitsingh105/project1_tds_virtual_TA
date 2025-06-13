import requests

session = requests.Session()
session.cookies.set("_t", "nv8IgZk8ZeydCwbC0DdAR8hXw9cQa9PqXxz%2FrUpn8Yddll%2F%2BGXIZ5HluztoUFyjk82HiVY8bT0pqVkHyKLTkcWc3YEkO831thd31CjarOTZbQo2xkSRw1doui%2FXTTEu2AUAEll%2BwB79WuvddpiqiEr9dVoUIauwYNLvmRfXOweZIEvb3cX42qO2f7vgYG%2BwFot0a8BHvIuUZIHNUpfhd8OtwJWKKoTPbI7FeGNn70KJ%2BuJQg0JhO%2FROO5JJz4QtRc4rYsg%2FlB4Y%2B6RQA%2FTDLXvKxggGHbt%2Bi18onSkY%2FvJAbj1HqGzVJ8TYV%2FSQpDkoyhhWOdQ%3D%3D--CCcA3hyKN%2B3aqeJf--lqFE818V9yZQJefUROcEEw%3D%3D", domain="discourse.onlinedegree.iitm.ac.in")

url = "https://discourse.onlinedegree.iitm.ac.in/c/courses/tds-kb/34.json"
response = session.get(url)

print(response.status_code)
print(response.text[:200])  # Show a snippet
