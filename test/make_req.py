
import unittest
import json
import requests

def recommendations():
    location = 'http://localhost:5000/api'
    device_id  = 'test_user1'
    
    rating1 = {'movie_id': '771028170', 'rating': '1'}
    rating2 = {'movie_id': '771321699', 'rating': '1'}
    rating3 = {'movie_id': '771363115', 'rating': '1'}
    rating4 = {'movie_id': '770680844', 'rating': '-1'}
    rating5 = {'movie_id': '771377017', 'rating': '-1'}
    rating6 = {'movie_id': '769959054', 'rating': '-1'}
    rating7 = {'movie_id': '771205997', 'rating': '1'}
    rating8 = {'movie_id': '770808377', 'rating': '1'}
    rating9 = {'movie_id': '771360513', 'rating': '1'}
    rating10 = {'movie_id': '771391173', 'rating': '1'}
    rating11 = {'movie_id': '770810295', 'rating': '1'}
    rating12 = {'movie_id': '771419323', 'rating': '1'}
    rating13 = {'movie_id': '771423060', 'rating': '-1'}
    rating14 = {'movie_id': '771236985', 'rating': '-1'}
    rating15 = {'movie_id': '771041155', 'rating': '-1'}
    rating16 = {'movie_id': '770671491', 'rating': '-1'}
    rating17 = {'movie_id': '460328477', 'rating': '1'}
    rating18 = {'movie_id': '771356295', 'rating': '1'}
    rating19 = {'movie_id': '770673038', 'rating': '1'}
    rating20 = {'movie_id': '112770454', 'rating': '1'}
    rating21 = {'movie_id': '10061', 'rating': '1'}
    rating22 = {'movie_id': '11560', 'rating': '1'}
    rating23 = {'movie_id': '10513', 'rating': '-1'}
    
    ratings = [
        rating1, rating2, rating3,
        rating4, rating5, rating6,
        rating7, rating8, rating9,
        rating10, rating11, rating12,
        rating13, rating14, rating15,
        rating16, rating17, rating18,
        rating19, rating20, rating21,
    ]
    ratings2 = [ rating22, rating23 ]

    
    data = {
        'ratings': json.dumps(ratings2),
        'device_id': device_id
    }
    res= requests.post(location, data=data)
    recommendations = json.loads(res.text)['recommendations']
    print recommendations
    return recommendations
    
class Test(unittest.TestCase):

    def test_recommendations(self):
        recs = recommendations()
        self.assertTrue(isinstance(recs, list))


if __name__ == '__main__':
    unittest.main()
