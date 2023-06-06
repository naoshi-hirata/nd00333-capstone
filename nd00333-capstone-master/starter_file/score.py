{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8de844ea",
   "metadata": {},
   "outputs": [],
   "source": [
    "import json\n",
    "import joblib\n",
    "import numpy as np\n",
    "from azureml.core.model import Model\n",
    "\n",
    "def init():\n",
    "    global model\n",
    "    model_path = Model.get_model_path('best_aml_model')  # モデルの登録名を指定\n",
    "    model = joblib.load(model_path)\n",
    "\n",
    "def run(raw_data):\n",
    "    data = json.loads(raw_data)['data']\n",
    "    data = np.array(data)\n",
    "    result = model.predict(data)\n",
    "    return result.tolist()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
