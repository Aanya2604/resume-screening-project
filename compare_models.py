import matplotlib.pyplot as plt
import numpy as np

# Results from classifier.py
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
before = [0.6499, 0.6581, 0.6499, 0.6327]   # Logistic Regression
after  = [0.7022, 0.7108, 0.7022, 0.6971]   # Linear SVM + class_weight="balanced"

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))

bars1 = ax.bar(x - width/2, before, width, label='Logistic Regression (Before)', color='#94A3B8')
bars2 = ax.bar(x + width/2, after, width, label='Linear SVM + Balancing (After)', color='#2F5496')

ax.set_ylabel('Score')
ax.set_title('Classifier Performance: Before vs After Improvement')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 0.85)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
ax.grid(axis='y', linestyle='--', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('classifier_comparison_chart.png', dpi=200)
print("Saved: classifier_comparison_chart.png")
plt.show()