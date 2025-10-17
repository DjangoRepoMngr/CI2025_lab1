# CI2025_lab1

This repo has one test notebook per problem (`test_problem1.ipynb`, `test_problem2.ipynb`, `test_problem3.ipynb`). Each test file defines the data and runs the algorithms. The file `solvers.py` contains the core algorithm logic (Hill Climbing, Tabu Search, ILS, SA), which the notebooks import and run.

## Implemented algorithms:

- Hill Climbing (naive): Greedy initialization followed by first-improvement single-item moves (including dropping items), always maintaining feasibility; stops at a local optimum.
- Tabu Search (single-item move with tenure): Explores the same single-item move neighborhood but forbids recent reverse moves via a short tabu tenure; uses aspiration to allow tabu moves that improve the global best.
- Iterated Local Search (ILS): Repeatedly applies the hill climbing local search; between runs it perturbs the current solution by dropping a few assigned items to escape local minima and keeps the best-so-far.
- Simulated Annealing (SA): Performs feasible single-item moves with probabilistic acceptance; always accepts improvements and sometimes accepts worse moves based on an exponential cooling schedule to escape local optima.


## Brief comparison on Problem 1

Tabu Search and Simulated Annealing found better solutions than Hill Climbing and ILS. Tabu was slower because it checks many candidate moves and keeps a tabu memory, but that extra search helped it improve further. SA was faster and still strong, since it sometimes accepts worse moves to escape traps early on. Hill Climbing and ILS were quick; ILS tries to shake the solution by dropping a few items and climbing again, but on this small case both still ended near the same local solution.

## Brief comparison on Problem 2

On the larger instance, Tabu Search found a clearly better solution than the other methods, but it took much longer (about 2.34 minutes). Hill Climbing and ILS were very fast yet stayed at a weaker local solution, showing limited exploration on bigger problems. Simulated Annealing improved over Hill Climbing/ILS with only a few seconds of runtime, trading a little time for better quality.

## Brief comparison on Problem 3

Hill Climbing was very fast but stopped at a local solution. ILS reached a similar result but took much longer (~236s). Simulated Annealing gave the best solution among the methods within a reasonable time (~50s) after lowering its parameters to keep it feasible. Tabu Search was not practical here: even after reducing its settings for several times it still ran 10–15 minutes on this machine, so I did not include a final result and gave up on it.


## Final conclusion

For these knapsack instances, simple Hill Climbing and ILS are fast but often stop at weaker local solutions. Methods that allow controlled exploration—Tabu Search and Simulated Annealing—tend to find better results; however, Tabu can be slow on large problems, while SA offers a good balance between solution quality and runtime. In practice, SA was the most reliable choice across sizes here, with Tabu worth using only when extra time is acceptable.

---

Note: I used  an AI assistance tool to help me with parts of the coding and documentation.