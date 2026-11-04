import json, collections, re, sys
R=[json.loads(l) for l in open(sys.argv[1])]
live=[r for r in R if not r["deleted"]]
print("total",len(R),"live",len(live),"malformed",sum(1 for r in live if len(r["options"])!=4 or not r["answer"]))
print(collections.Counter(r["file"] for r in live))
withref=[r for r in live if r["adn_refs"]]; print("with refs",len(withref))
H=json.load(open(sys.argv[2]))
ids={re.sub(r"^Volume [IV]+ / ","",h["id"]) for h in H}
refs=collections.Counter(x for r in withref for x in r["adn_refs"])
hit=sum(1 for x in refs if x in ids); print("distinct refs",len(refs),"found",hit)
q=sum(1 for r in withref if all(x in ids for x in r["adn_refs"])); print("questions all refs found",q,"/",len(withref))
miss=[x for x in refs if x not in ids]; print("missing:", sorted(miss))
