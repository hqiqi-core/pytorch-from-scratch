class Animal:

    def eat(self):
        print("Animal is eating")

class Dog(Animal):   # 继承 Dog 自动拥有 Animal 的方法
   def bark(self):   #扩展
        print("Woof!")

dog = Dog()
dog.eat()
dog.bark()