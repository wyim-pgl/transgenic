import sys


stats = sys.argv[1]
#stats = "TestSetPerformanceAnalysis/NT_transgenic_Gen10G_6144nt_E4-AgroSegmentNT_Epoch6_6144nt_restart_codons/TAIR10_post.stats"
print(stats)

with open(stats, "r") as infile:
	lines = infile.readlines()

values = []
values.append(lines[7].split()[6])
values.append(lines[7].split()[4])
values.append(lines[5].split()[6])
values.append(lines[5].split()[4])
values.append(lines[10].split()[2])
values.append(lines[10].split()[4])
values.append(lines[11].split()[2])
values.append(lines[11].split()[4])
values.append(lines[12].split()[2])
values.append(lines[12].split()[4])
values.append(lines[13].split()[3])
values.append(lines[13].split()[5])
values.append(lines[14].split()[2])
values.append(lines[14].split()[4])
values.append(lines[15].split()[2])
values.append(lines[15].split()[4])

print(",".join(values))