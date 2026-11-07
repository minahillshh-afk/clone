# My study notes: AI search, compilers and cloud

These are notes from my exam preparation in July 2026, written in my own words.

## AI course: search and optimization
Uninformed search explores the problem blindly, while informed search uses a heuristic to guess which state is closer to the goal. Best-First Search always expands the state that looks best by the heuristic alone. Beam Search keeps only the best few states at each level, which saves memory but can miss the optimal path. IDDFS repeats depth-limited depth-first search with a growing limit, so it uses little memory and still finds the shallowest goal. IDA* does the same idea with a cost limit instead of a depth limit.

Hill Climbing moves to the best neighbouring state and stops when no neighbour is better. Its weakness is that it can get stuck in local maxima, plateaus or ridges. For two-player games I studied Alpha-Beta Pruning, which gives the same answer as minimax but skips branches that cannot change the final decision. I also studied constraint satisfaction problems (CSPs), the PEAS framework for describing an intelligent agent (Performance measure, Environment, Actuators, Sensors), and genetic algorithms, which evolve solutions through selection, crossover and mutation.

## Compiler construction: parsing
LL(1) parsers read the input left to right and build a leftmost derivation using one lookahead symbol. For that to work, left recursion has to be removed from the grammar. CLR(1) parsers use canonical sets of items with lookahead symbols, GOTO transitions and a parsing table. LALR merges CLR(1) states that have the same core to make the table smaller. After parsing, the compiler builds a parse tree or an abstract syntax tree (AST) and generates three-address code (TAC), which can be stored as quadruples, triples or indirect triples. I also practised converting infix expressions to postfix.

## Cloud computing
I studied mobile cloud computing and offloading methods, and AWS services: S3 for object storage, EC2 for virtual servers, RDS and Aurora for relational databases, DynamoDB for NoSQL, ElastiCache for caching and Redshift for data warehousing. Exam questions were scenario-based, where I had to pick the right service for an architecture. I also covered AWS Bedrock and generative AI foundations.
