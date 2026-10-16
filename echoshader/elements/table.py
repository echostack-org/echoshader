import holoviews
import pandas
import xarray


def table(
    MVBS_ds: xarray.Dataset,
):
    """
    Create a table of summary statistics for Sv.

    Parameters
    ----------
    MVBS_ds : xarray.Dataset
        Dataset containing the ``Sv`` variable.

    Returns
    -------
    holoviews.Table
        Summary statistics for the complete dataset and each channel.
    """
    obj_df_sum = MVBS_ds.Sv.to_dataframe()

    skew_sum = obj_df_sum["Sv"].skew()
    kurt_sum = obj_df_sum["Sv"].kurt()

    obj_desc = obj_df_sum.describe().reset_index()[["index", "Sv"]].rename(columns={"Sv": "Sum"})

    obj_desc.loc[len(obj_desc)] = [
        "skew",
        skew_sum,
    ]

    obj_desc.loc[len(obj_desc)] = [
        "kurtosis",
        kurt_sum,
    ]

    for channel in MVBS_ds.channel.values:
        obj_df_channel = MVBS_ds.sel(channel=channel).Sv.to_dataframe()

        skew_channel = obj_df_channel["Sv"].skew()
        kurt_channel = obj_df_channel["Sv"].kurt()

        obj_df_channel = (
            obj_df_channel.describe().reset_index()[["index", "Sv"]].rename(columns={"Sv": channel})
        )

        obj_df_channel.loc[len(obj_df_channel)] = [
            "skew",
            skew_channel,
        ]

        obj_df_channel.loc[len(obj_df_channel)] = [
            "kurtosis",
            kurt_channel,
        ]

        obj_desc = pandas.merge(
            obj_desc,
            obj_df_channel,
            on="index",
        )

    return holoviews.Table(obj_desc)
