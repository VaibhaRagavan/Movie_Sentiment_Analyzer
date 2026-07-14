from flask import Flask,request,render_template,jsonify,redirect
import requests,numpy as np
from transformers import pipeline
import torch
from models import pipeline as sentiment_pipeline 
import os
from dotenv import load_dotenv
load_dotenv()
key=os.environ.get("TMDB_API_KEY")
app=Flask(__name__)
@app.route("/",methods=["GET"])
def main():
    return render_template("index.html")

@app.route("/get_movie", methods=["POST"])
def movie_detail():
    if request.method =="POST":
        data=request.get_json()
        movieid=data["id"]
        key=key
        url=f"https://api.themoviedb.org/3/movie/{movieid}?api_key={key}"
        response=requests.get(url)
        data=response.json()
        Movie_Name=data["original_title"]
        Runtime=data["runtime"]
        Overview=data["overview"]
        Rating=data["vote_average"]

        review_url=f"https://api.themoviedb.org/3/movie/{movieid}/reviews?api_key={key}"
        response_review=requests.get(review_url)
        review_data=response_review.json()
        review_results=review_data["results"]
        reviews=[]
        if review_results:
            for item in review_results[:5]:
                rv=item["content"]
                reviews.append(rv)
        else:
            reviews=[]
        ##sentiment analysis using pipeline
        review_score=sentiment_pipeline.predict(reviews)
        
        #overview sentiment analysis
        overview_score=sentiment_pipeline.overview_analysis(Overview)
        #verdict is mean of review score,rating and score for overview
        verdict_score=np.mean([overview_score,review_score,Rating/10])
       
        verdict=""
        if verdict_score >=0.7:
            verdict="Worth Watching"
        elif verdict_score >=0.4:
            verdict="Watchable"
        else:
            verdict="Not Worth to Watch"
        

        details={
            "movie_name":Movie_Name,
            "runtime":Runtime,
            "overview":Overview,
            "rating":Rating,
            "review_score":review_score*10,
            "verdict":verdict
        }
        
        return jsonify(data=details)
    
if __name__ =="__main__":
    app.run(debug=True)
