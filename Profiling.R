# Title     : TODO
# Objective : TODO
# Created by: jeanclaudelemoyne
# Created on: 7/16/18
ic_ship_qty <- read.csv('/Users/jeanclaudelemoyne/work/Data/Unilever/ICDressing/train_ic_ship_qty.csv')
print (dim(ic_ship_qty))
tship <- ic_ship_qty$ship
nzship <- c()
m <- length(tship)
print(m)
k <- 0
for (value in tship) {
    if (value <= 0) {
        k <- k + 1
        nzship[k] <- value
    }
}
print (length(nzship))
print (k)
minship <- min(nzship)
print (cat('min: ', minship, "\n"))
print ('... done')
