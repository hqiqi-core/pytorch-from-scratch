class Student:
    def __init__(self,name,age):
        self.name =name
        self.age = age
    def introduce(self):
        print(f"My name is {self.name}, I am {self.age} years old.")
    



student1 = Student("HQQ", 24)
student2 = Student("Alice", 23)
print(student1.name)
print(student1.age)

print(student2.name)
print(student2.age)
student1.introduce()
student2.introduce()