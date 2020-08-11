FILES = system("ls -1 ./raw_proc_*_log.txt")
LABEL = system("ls -1 ./raw_proc_*_log.txt | sed -e 's/\_//' -e 's/.txt//'")
TITLE = 'pong'

set xtic auto
set ytic auto
set xlabel 'time (s)'
set ylabel 'score'
set title TITLE
set grid ytics lt 0 lw 1 lc rgb "#bbbbbb"
set grid xtics lt 0 lw 1 lc rgb "#bbbbbb"
plot for [i=i=1:words(FILES)] word(FILES,i) u 2:14 pt 0.5 ps 0.5 (i) title word(LABEL,i) noenhanced
fname_ps = sprintf("./plots/%s.ps",TITLE)
fname_png = sprintf("./plots/%s.png",TITLE)
fname_g = sprintf("./plots/%s.g",TITLE)
set term postscript
set output fname_ps
replot
