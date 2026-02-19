import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

def trouve_pics(signal, tolerance: int = 1, height_threshold: list = None, distance: list = None,
                left_base_width: list = None, right_base_width: list = None, total_base_width: list = None,
                threshold_width: list = None, percent_height_at_threshold_width: float = 50,
                left_prominence: list = None, right_prominence: list = None, prominence: list = None,
                plot: bool = False, pics_and_param: bool = False):
    """ Fonction équivalente à find_peaks de Noah Keraudren. La fonction possède des paramètres plus précis et poussés
    que la fonction find_peaks classique.

    :param signal: Signal avec les pics à trouver
    :param tolerance: Paramètre permettant d'éliminer les tout petits artefacts. Avec une valeur de 1, tous les pics
    seront détectés. Plus la valeur est haute, plus les artefacts de petite taille et largeur disparaitront. Ne pas
    fournir de valeurs plus grandes qu'un demi-pic d'intérêt.
    :param height_threshold: Seuil de hauteur absolue (format : [min, max] ou [None, max] ou [min, None])
    :param distance: Seuil de distance inter-pics en index (format : [min, max] ou [None, max] ou [min, None])
    :param left_base_width: Seuil de largeur du pic à gauche (format : [min, max] ou [None, max] ou [min, None])
    :param right_base_width: Seuil de largeur du pic à droite (format : [min, max] ou [None, max] ou [min, None])
    :param total_base_width: Seuil de largeur totale du pic (format : [min, max] ou [None, max] ou [min, None])
    :param threshold_width: Seuil de largeur du pic à la hauteur seuil définie par 'percent_height_at_threshold_width'.
    (format : [min, max] ou [None, max] ou [min, None])
    :param percent_height_at_threshold_width: Valeur strictement entre 0 et 100. Correspond au pourcentage de la hauteur
    de proéminence minimale du pic (gauche ou droite), auquel sera placé le seuil pour la 'threshold_width'
    :param left_prominence: Seuil de proéminence à gauche (format : [min, max] ou [None, max] ou [min, None])
    :param right_prominence: Seuil de proéminence à droite (format : [min, max] ou [None, max] ou [min, None])
    :param prominence: Seuil pour la proéminence minimale (gauche ou droite) (format : [min, max] ou [None, max] ou
    [min, None])
    :param plot: Si True : affiche la courbe de signal avec l'ensemble des points, et les points valides
    :param pics_and_param: Si True : retourne un pd.DataFrame contenant les pics valides, et leurs paramètres. Si False,
    retourne un pd.DataFrame contenant les 'index' et 'value' de chaque pic uniquement.
    :return: Dépend de 'pics_and_params'
    """
    def search_threshold_validated_values(pics_df: pd.DataFrame, parameter: str, val_parameter: list):
        """ Fonction de comparaison des seuils. Sert à comparer une colonne de paramètres avec les seuils définis dans
        une liste à deux éléments. Les listes sont au format [min,max] et peuvent ne contenir qu'une valeur accompagnée
        d'un None si l'on ne souhaite qu'un min ou un max.

        :param pics_df: pd.DataFrame contenant en colonnes : {'index','value', ... et tous les paramètres de
        caractérisation du pic}.
        :param parameter: Nom du paramètre dont nous souhaitons comparer les seuils.
        :param val_parameter: Les valeurs de seuil du paramètre d'intérêt. Prends la forme [min, max] ou [None, max],
        ou [min, None]
        :return: Retourne un pd.DataFrame contenant des booléens spécifiant si les critères de seuil sont remplis par
        chaque pic
        """
        if val_parameter != None and len(val_parameter) > 2:  # vérification de la dimension de la liste
            raise ValueError('Parameter list dimension superior to maximal size of 2')
        elif val_parameter == None:
            param_df = pd.DataFrame({parameter: np.full(len(pics_df), 'True')})
        elif val_parameter != None:
            if val_parameter[0] != None and val_parameter[1] != None:  # double seuil
                param_df = pd.DataFrame(
                    {parameter: (pics_df[parameter] > val_parameter[0]) & (pics_df[parameter] < val_parameter[1])})
            elif val_parameter[0] != None and val_parameter[1] == None:  # seulement seuil minimal
                param_df = pd.DataFrame({parameter: pics_df[parameter] > val_parameter[0]})
            elif val_parameter[0] == None and val_parameter[1] != None:  # seulement seuil maximal
                param_df = pd.DataFrame({parameter: pics_df[parameter] < val_parameter[1]})
        return param_df

    signal = np.array(signal) # transformation du signal en np.array
    indices = []
    values = []

    # Recherche des pics comme des maximums en trois point [i-tolérance ; i ; i+tolérance].
    # L'augmentation de la tolérance permet d'éviter les artefacts sur le signal, mais créé des plages de pics aux
    # endroits des vrais pics. Ces plages sont rafinées à l'étape suivante.
    for i in range(tolerance, len(signal) - tolerance):
        val = signal[i]
        if signal[i - tolerance] < val > signal[i + tolerance]:
            indices.append(i)
            values.append(val)
    temp_pics_df = pd.DataFrame({'index': indices, 'value': values})


    # Rafinement des pics pour éviter les successions de pics à cause d'une tolérance haute.
    pics_df = pd.DataFrame({'index': [], 'value': []})
    for i in range(len(temp_pics_df)):  # Pour tous les pics trouvés avec tolérance
        # Si nous sommes au premier pic, nous stockons la première valeur comme pic local, et suite est False
        if i == 0:
            local_pic = temp_pics_df.iloc[0, :]
        # Si nous sommes sur les pics après le premier ...
        if i > 0:
            index_pic = temp_pics_df['index'][i]
            index_prev_pic = temp_pics_df['index'][i - 1]
            # Nous vérifions que c'est bien une suite et que nous ne sommes pas au dernier indice ...
            if (index_pic - 1) == index_prev_pic and i + 1 != len(temp_pics_df):
                value_pic = temp_pics_df['value'][i]
                value_local_pic = local_pic[1]
                # Nous calculons la différence entre le pic local stocké, et le pic sur lequel nous sommes
                diff_pics = value_local_pic - value_pic
                # Si le pic local stocké est plus grand, nous le gardons
                if diff_pics >= 0:
                    local_pic = local_pic
                # Sinon si le pic actuel est plus grand, nous le stockons
                elif diff_pics < 0:
                    local_pic = temp_pics_df.iloc[i, :]
            # Sinon, nous stockons la valeur de pic local déterminée précédemment dans le pics_df
            else:
                new_pic = pd.DataFrame({'index': [int(local_pic['index'])], 'value': [local_pic['value']]})
                pics_df = pd.concat([pics_df, new_pic], axis=0, ignore_index=True)
                local_pic = temp_pics_df.iloc[i, :]
    pics_df['index'] = pics_df['index'].astype(int)

    # Recherche des largeurs à gauche et à droite, et de proéminence pour chaque pic
    pics_left_widths = []
    pics_right_widths = []
    prominence_left = []
    prominence_right = []

    for i,pic_index in enumerate(pics_df['index']):
        left_pic_attained = False
        right_pic_attained = False
        # Tant que nous ne sommes pas arrivés au pic suivant à gauche ou à droite, nous recherchons un minimum

        # Recherche à gauche
        min_index = pic_index-1
        min_val = signal[min_index]
        test_index = min_index
        while left_pic_attained == False:
            if i == 0 :
                if test_index == 0 :
                    left_pic_attained = True
            elif i > 0 :
                if test_index-1 == pics_df['index'][i-1]:
                    left_pic_attained = True
            if signal[test_index] < min_val:
                min_index = test_index
                min_val = signal[test_index]
            test_index = test_index - 1
        width_left = pic_index-min_index

        # Recherche à droite
        min_index = pic_index + 1
        min_val = signal[min_index]
        test_index = min_index
        while right_pic_attained == False:
            if i == len(pics_df)-1:
                if test_index == len(signal)-1:
                    right_pic_attained = True
            elif i < len(pics_df)-1:
                if test_index + 1 == pics_df['index'].iloc[i + 1] :
                    right_pic_attained = True
            if signal[test_index] < min_val:
                min_index = test_index
                min_val = signal[test_index]
            test_index = test_index + 1
        width_right = min_index - pic_index

        # Création et ajout des paramètres des pics au df
        pics_left_widths.append(width_left - 1)
        pics_right_widths.append(width_right - 1)
        pic_value = signal[pic_index]
        prominence_l = pic_value - signal[pic_index - width_left]
        prominence_r = pic_value - signal[pic_index + width_right]
        prominence_left.append(prominence_l)
        prominence_right.append(prominence_r)

    pics_df['left_base_width'] = pics_left_widths
    pics_df['right_base_width'] = pics_right_widths
    pics_df['base_width'] = [a + b for a, b in zip(pics_left_widths, pics_right_widths)]
    pics_df['prominence_left'] = prominence_left
    pics_df['prominence_right'] = prominence_right
    pics_df['max_prominence'] = [max(a, b) for a, b in zip(prominence_left, prominence_right)]
    pics_df['min_prominence'] = [min(a, b) for a, b in zip(prominence_left, prominence_right)]

    # Recherche de la largeur à un certain % de la proéminence min du pic
    width_at_threshold = []
    for i, pic_index in enumerate(pics_df['index']):
        side_of_min_prominence = np.argmin([pics_df['prominence_left'][i], pics_df['prominence_right'][i]])
        if side_of_min_prominence == 0:  # prominence_left
            short_base_index = pic_index - pics_df['left_base_width'][i]
            short_base_value = signal[short_base_index]
            short_prominence = pics_df['prominence_left'][i]

        elif side_of_min_prominence == 1:  # prominence_right
            short_base_index = pic_index + pics_df['right_base_width'][i]
            short_base_value = signal[short_base_index]
            short_prominence = pics_df['prominence_right'][i]

        # Définition de la hauteur de seuil en % de la hauteur de la proéminence minimale
        threshold_height = short_base_value + short_prominence * (percent_height_at_threshold_width / 100)

        # Recherche de la valeur au-dessus du seuil à gauche
        current_left_index = pic_index
        left_base_index = pic_index - pics_df['left_base_width'][i]
        left_threshold_attained = False
        while left_threshold_attained == False :
            # Si current_left_index -1 n'est pas l'index de la base gauche
            if current_left_index - 1 > left_base_index:
                # Si current_left_index -1 est inférieur au seuil
                if signal[current_left_index - 1] < threshold_height :
                    left_sup_threshold = current_left_index
                    left_threshold_attained = True
                else :
                    current_left_index = current_left_index-1 # incrément de valeur vers la gauche

            # Si current_left_index -1 est l'index de la base gauche
            elif current_left_index - 1 <= left_base_index:
                # Si valeur à current_left_index -1 est inférieure au seuil
                if signal[current_left_index - 1] < threshold_height:
                    left_sup_threshold = left_base_index
                else:
                    left_sup_threshold = current_left_index
                left_threshold_attained = True

        # Recherche de la valeur au-dessus du seuil à droite
        current_right_index = pic_index
        right_base_index = pic_index + pics_df['right_base_width'][i]
        right_threshold_attained = False
        while right_threshold_attained == False:
            # Si current_right_index +1 n'est pas l'index de la base droite
            if current_right_index + 1 < right_base_index:
                # Si current right_index +1 est inférieur au seuil
                if signal[current_right_index +1] < threshold_height:
                    right_sup_threshold = current_right_index
                    right_threshold_attained = True
                else:
                    current_right_index = current_right_index + 1  # incrément de valeur vers la droite

            # Si current_right_index + 1 est l'index de la base droite
            elif current_right_index + 1 >= right_base_index:
                # Si valeur à current_right_index +1 est inférieure au seuil
                if signal[current_right_index + 1] < threshold_height:
                    right_sup_threshold = right_base_index
                else:
                    right_sup_threshold = current_right_index
                right_threshold_attained = True

        width = right_sup_threshold - left_sup_threshold +1
        width_at_threshold.append(width)
    pics_df['width_at_threshold'] = width_at_threshold

    # Création du dataframe contenant les critères de validation de chaque cycle
    parameters_dict = {'value': height_threshold,
                       'left_base_width': left_base_width,
                       'right_base_width': right_base_width,
                       'base_width': total_base_width,
                       'prominence_left': left_prominence,
                       'prominence_right': right_prominence,
                       'min_prominence': prominence,
                       'width_at_threshold': threshold_width, }
    validation_criterions_df = pd.DataFrame()

    for parameter, val_parameter in parameters_dict.items():
        param_list = search_threshold_validated_values(pics_df, parameter, val_parameter)
        validation_criterions_df = pd.concat([validation_criterions_df, param_list], axis=1)

    # Sélection des pics remplissant tous les critères attendus
    validated_indexes = validation_criterions_df.all(axis=1)
    validated_pics = pics_df[validated_indexes].copy().reset_index(drop=True)

    # Ajout du critère de distance entre deux pics.
    # Départ au premier pic, puis vérification de chaque pic et de sa distance au précédent pic valide.
    if distance != None:
        list_distances = []
        list_validation = [True]
        for i, pic_index in enumerate(validated_pics['index']):
            if i > 0:
                prev_pic_num = 1
                prev_pic_validated = (list_validation[i - prev_pic_num])
                while prev_pic_validated == False:
                    prev_pic_num = prev_pic_num + 1
                    prev_pic_validated = (list_validation[i - prev_pic_num])

                dist = pic_index - validated_pics['index'][i - prev_pic_num]
                if distance[0] != None and distance[1] != None:  # double seuil
                    if distance[0] < dist < distance[1]:
                        list_validation.append(True)
                    else:
                        list_validation.append(False)
                elif distance[0] != None and distance[1] == None:  # seulement seuil minimal
                    if distance[0] < dist:
                        list_validation.append(True)
                    else:
                        list_validation.append(False)
                elif distance[0] == None and distance[1] != None:  # seulement seuil maximal
                    if dist < distance[1]:
                        list_validation.append(True)
                    else:
                        list_validation.append(False)

        validated_pics = validated_pics[list_validation]

    # Figure de vérification
    if plot is True:
        # Création de la figure
        fig = go.Figure()
        # Courbe du signal
        fig.add_trace(go.Scatter(
            x=np.arange(len(signal)),
            y=signal,
            mode='lines',
            name='Signal',
            line=dict(color='blue')
        ))
        # Points de tous les pics
        fig.add_trace(go.Scatter(
            x=pics_df['index'],
            y=pics_df['value'],
            mode='markers',
            name='Pics',
            marker=dict(color='red', size=8, symbol='circle')
        ))
        # Points des pics validés
        fig.add_trace(go.Scatter(
            x=validated_pics['index'],
            y=validated_pics['value'],
            mode='markers',
            name='Pics valides',
            marker=dict(color='green', size=10, symbol='circle')
        ))
        fig.update_layout(
            title='Signal avec pics détectés',
            xaxis_title='Index',
            yaxis_title='Amplitude',
            template='plotly_white'
        )
        fig.show()

    # Si l'on souhaite un df avec les pics et tous les paramètres des pics validés
    if pics_and_param == True:
        return validated_pics
    # Si l'on souhaite un df avec 'index' et 'value' des pics uniquement (par défaut)
    elif pics_and_param == False:
        return validated_pics[['index', 'value']]