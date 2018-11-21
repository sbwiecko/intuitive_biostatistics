from scipy import stats

a=.8
b=.9
c=.95
d=.975
e=.99
f=.995

print("Critical values of Student's t distribution with df degrees of freedom")
print("Probability less than the critical value (t(1-alpha, df))")

print(f"df\t{a:4.3f}\t{b:4.3f}\t{c:4.3f}\t{d:4.3f}\t{e:4.3f}\t{f:4.3f}")
print("-------------------------------------------------------", end='')
for df in range(1, 16):
	print(f"\n{df}", end="")
	for alpha in [a,b,c,d,e,f]:
		print(f"\t{stats.t(df=df).ppf((1+alpha)/2):4.3f}", end="")
     
with open('table_t_2-tailed.txt', 'w') as file:
	print("Critical values of Student's t distribution with df degrees \
	   of freedom", file=file)
	print("Probability less than the critical value (t(1-alpha, df))", file=file)
	
	print(f"df\t{a:4.3f}\t{b:4.3f}\t{c:4.3f}\t{d:4.3f}\t{e:4.3f}\t{f:4.3f}", file=file)
	print("-------------------------------------------------------", end='', file=file)
	for df in range(1, 16):
		print(f"\n{df}", end="", file=file)
		for alpha in [a,b,c,d,e,f]:
			print(f"\t{stats.t(df=df).ppf((1+alpha)/2):4.3f}", end="", file=file)
