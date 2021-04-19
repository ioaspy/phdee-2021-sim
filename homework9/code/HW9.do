import excel "\\prism.nas.gatech.edu\ispyrou3\vlab\documents\YL_paper\environ\fishbycatchupdated.xlsx", sheet("in") firstrow
(111 vars, 50 obs)

reshape long shrimp salmon bycatch, i(firm) j(month)

*question 1

bys month treated: egen new=total(bycatch)
gen Control_group=new if treated==0
gen Treated_group=new if treated==1
graph twoway line Control_group Treated_group month
graph export mygraph.eps

*question 2

generate treat=treated
replace treat=0 if month<=12
replace treat=1 if month>24


xtset firm month
xtreg bycatch treat i.month shrimp salmon, fe vce(cluster firm)
outreg2 using Output.doc

*question 3
reghdfe bycatch shrimp salmon treat, absorb(firm month) vce(cluster firm#month)

*question 5
twowayfeweights bycatch firm month treat, type(feTR) controls(shrimp salmon)

*question 7

drop if month > 24
twowayfeweights bycatch firm month treat, type(feTR) controls(shrimp salmon)