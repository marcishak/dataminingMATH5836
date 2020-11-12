#Source:https://www.r-bloggers.com/2015/04/id3-classification-using-data-tree/

IsPure <- function(data) {
  length(unique(data[,ncol(data)])) == 1
}
Entropy <- function( vls ) {
  res <- vls/sum(vls) * log2(vls/sum(vls))
  res[vls == 0] <- 0
  -sum(res) }


InformationGain <- function( tble ) {
  tble <- as.data.frame.matrix(tble)
  entropyBefore <- Entropy(colSums(tble))
  s <- rowSums(tble)
  entropyAfter <- sum (s / sum(s) * apply(tble, MARGIN = 1, FUN = Entropy ))
  informationGain <- entropyBefore - entropyAfter
  return (informationGain)
}


TrainID3 <- function(node, data) {
  
  node$obsCount <- nrow(data)
  
  #if the data-set is pure (e.g. all toxic), then
  if (IsPure(data)) {
    #construct a leaf having the name of the pure feature (e.g. 'toxic')
    child <- node$AddChild(unique(data[,ncol(data)]))
    node$feature <- tail(names(data), 1)
    child$obsCount <- nrow(data)
    child$feature <- ''
  } else {
    #chose the feature with the highest information gain (e.g. 'color')
    ig <- sapply(colnames(data)[-ncol(data)], 
                 function(x) InformationGain(
                   table(data[,x], data[,ncol(data)])
                 )
    )
    feature <- names(ig)[ig == max(ig)][1]
    
    node$feature <- feature
    
    #take the subset of the data-set having that feature value
    childObs <- split(data[,!(names(data) %in% feature)], data[,feature], drop = TRUE)
    
    for(i in 1:length(childObs)) {
      #construct a child having the name of that feature value (e.g. 'red')
      child <- node$AddChild(names(childObs)[i])
      
      #call the algorithm recursively on the child and the subset      
      TrainID3(child, childObs[[i]])
    }
    
  }
  
  
  
}


#------------------------------------  test above functions

#If a set contains 5 poisonous and 5 edible mushrooms, 
#the entropy becomes 1, as the purity is at its lowest:

#Entropy(c(5, 5))
#We can plot the entropy as a function of the number of edible mushrooms
# in a set of, say, 100 mushrooms:

entropy <- function(edible) Entropy(c(edible, 100 - edible))
entropy <- Vectorize(entropy)
curve( entropy, from = 0, to = 100, xname = 'edible')

#For example, using the mushroom data set:

library(data.tree) # this is not installed in Ed - try in your PC

data(mushroom)
tble <- table(mushroom[,c('color', 'edibility')])
print(tble)

tree <- Node$new("mushroom")

TrainID3(tree, mushroom)
print(tree, "feature", "obsCount")

Predict <- function(tree, features) {
  if (tree$children[[1]]$isLeaf) return (tree$children[[1]]$name)
  child <- tree$children[[features[[tree$feature]]]]
  return ( Predict(child, features))
}

Predict(tree, c(color = 'red', 
                size = 'large', 
                points = 'yes')
)

