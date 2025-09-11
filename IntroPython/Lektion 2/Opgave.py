#2-1
message = "Dette er nemt"
print(message)

#2-2

message = "Dette er stadig nemt"
print(message)

#2-3

personal_message = "Hello Eric"

#2-4

name = "lars"

print(name.title())
print(name.upper())
print(name.lower())

#2-5 / 2-6

nameFamous = "50cent"
quote = f"{nameFamous} once said sleep is for those who are broke i dont sleep"

print(quote)

#2-8

print(5+3)
print(4*2)
print(16/2)
print(16-8)

fav_number = 8

fav_number_text = f"{fav_number} er fucking sejt"
print(fav_number_text)

#3-1 / 3-2
navne = ["Lizzy","Kis","Augnete","Melissa"]
print(navne)

print(f"Fuck {navne[0]},Marry {navne[1]}, Kill {navne[2]}")

#3-3

transport =["Tog", "Bus", "Bil"]

print(f"Jeg fucking hader {transport[1]} og {transport[0]} men at køre i {transport[2]} er fed nok")

#3-4
guest = ["person1", "person2", "person3"]
print(f"Hej {guest[0]} vil du drikke 1.000.000 øl med mig på mandag?")
print(f"Hej {guest[1]} vil du drikke 1.000.000 øl med mig på mandag?")
print(f"Hej {guest[2]} vil du drikke 1.000.000 øl med mig på mandag?")

#3-5
print("ej fuck person2 kan ik komme miv")
guest.remove("person2")
guest.append("person4")
print(f"Hej {guest[2]} vil du drikke 1.000.000 øl med mig på mandag?")

#3-6
guest.insert(1,"PERSON")
print(guest)
#3-7
print("mor siger jeg kun må invitere 2")
guest.pop()
guest.pop()
print(guest)
print(f"jeg må kun invitere {guest[1]} og {guest[0]}")
del guest[0]
del guest[0]
print(guest)

#3-8

drømmeSteder = ["New York", "Japan", "Bali", "Frankrig"]
print(drømmeSteder)
print(sorted(drømmeSteder))
print(drømmeSteder)
print(sorted(drømmeSteder,reverse=True))
print(drømmeSteder)
drømmeSteder.reverse()
print(drømmeSteder)