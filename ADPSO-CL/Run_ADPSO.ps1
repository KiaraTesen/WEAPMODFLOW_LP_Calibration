param($p1, $p2)

$total_iteration = [int]($p1)
$final_iteration = [int]($p2)
$vm = [int]((hostname) -replace '\D+(\d+)','$1')

python Methodology_ADPSO_CL.py 10.0.0.11:8888 $total_iteration $final_iteration $vm