from flask import Flask,redirect,url_for,request,jsonify,render_template
from Summarizer import Summarizer
from RBMSummarizer import RBM_summarizer
app = Flask(__name__)



@app.route('/textsummary')
def get_text_summary():
    act_text = request.args.get('text','',type=str)
    choice = request.args.get('choice',1,type=int)
    s = Summarizer(act_text)
    rbm_s =RBM_summarizer(act_text)

    if choice==1:
        summ_text = s.word_counts_based_summary()

    elif choice==2:
        # summ_text =" You choose option 1"
        summ_text = s.get_summary_using_dependency_parsing()
    elif choice==3:
        # summ_text = " You choose option 2"
        summ_text = s.pagerank_summarizer()
    elif choice==4:

        summ_text=rbm_s.get_summary()
    else:
        pass

    return jsonify(result=summ_text)

@app.route('/')
def index():
    return render_template('text_summary.html')



# @app.route('/text_summary',methods=['POST','GET'])

# def add_numbers():
#     a = request.args.get('a', 0, type=int)
#     b = request.args.get('b', 0, type=int)
#     return jsonify(result=a + b)
# def get_summary():
#     if request.method == 'POST':
#         news_text = request.form['actual_text']
#         summary_choice = request.form['summary_type']
#
#         s = Summarizer(news_text)
#
#         if summary_choice=='1':
#             summ_text = s.get_summary_using_dependency_parsing()
#         elif summary_choice=='2':
#             summ_text = s.pagerank_summarizer()
#         elif summary_choice=='3':
#             summ_text="None"
#         else:
#             pass
#
#         return jsonify(result=summ_text)
#     else:
#         pass


if __name__ == '__main__':
   app.run(debug =True)