print("========== Dataset Statistics ==========")
with open("python_basics/data/labels.txt","r") as f :
    labels = [line.strip().lower() for line in f if line.strip()]
    total = len(labels)
print("Total samples:",total)
counts = {}
for label in labels:
    counts[label] = counts.get(label,0)+1
class_num = len(counts)
print("Number of classes:",class_num)

print("Class distribution:")
for label,count in counts.items():
    ratio = count / total
    print(f"{label}:{count} ({ratio:.2%})")