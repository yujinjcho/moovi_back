{-# LANGUAGE OverloadedStrings #-}

import Prelude hiding (lookup, readFile)
import System.Environment (getArgs)
import Data.Aeson
import Data.List (foldr, sortBy)
import Data.Function (on)
import Data.Map (Map, empty, lookup, insertWith, toList)
import Data.ByteString.Lazy (readFile, split)
import Data.ByteString.Internal (c2w)

data Rating = Positive | Negative deriving (Show)

instance FromJSON Rating where
  parseJSON (Number 1)    = pure Positive
  parseJSON (Number (-1)) = pure Negative

data Review = Review
  { movie  :: String
  , critic :: String
  , rating :: Rating
  } deriving (Show)

instance FromJSON Review where
  parseJSON (Object o) =
    Review <$> o .: "movie"
           <*> o .: "critic"
           <*> o .: "rating"

data Critic = Critic
  { positives :: [String]
  , negatives :: [String]
  , total     :: Int
  } deriving (Show)

instance Monoid Critic where
  mempty = Critic [] [] 0
  Critic p1 n1 t1 `mappend` Critic p2 n2 t2 = Critic (p1 ++ p2) (n1 ++ n2) (t1 + t2)

fromReview :: Review -> Critic
fromReview (Review m _ Positive) = Critic [m] [] 1
fromReview (Review m _ Negative) = Critic [] [m] 1

byCritic :: [Review] -> Map String Critic
byCritic = foldr (\r -> insertWith mappend (critic r) (fromReview r)) empty

reviewCounts :: [Review] -> Map String Int
reviewCounts = foldr (\r -> insertWith (+) (critic r) 1) empty

save :: Map String Critic -> String -> IO ()
save critics name =
  let saveKind kind f = writeFile (name ++ "_" ++ kind) . unlines . f
  in do
    let Just critic = lookup name critics
    saveKind "positives" positives critic
    saveKind "negatives" negatives critic

saveAll :: [String] -> [Review] -> IO ()
saveAll names reviews =
  let saveCritics = save $ byCritic reviews
  in mapM_ saveCritics names

top :: Int -> [Review] -> IO ()
top n reviews =
  let sorted = sortBy (compare `on` (negate . snd)) . toList . reviewCounts $ reviews
  in mapM_ (\(c, r) -> putStrLn $ c ++ " " ++ show r) $ take n sorted

main = do
  [file, action, arg] <- getArgs
  contents <- readFile file
  let json = init $ split (c2w '\n') contents
  let Just reviews = mapM (\r -> decode r :: Maybe Review) json

  case action of
    "save" -> saveAll (words arg) reviews
    "top"  -> top (read arg) reviews
