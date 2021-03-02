#   Copyright (c) 2021 Robert Bosch GmbH
#   Author: Fabian Otto
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Affero General Public License as published
#   by the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Affero General Public License for more details.
#
#   You should have received a copy of the GNU Affero General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.

from cox.store import Store, _clean_dict


class CustomStore(Store):
    """
    The following class is derived from cox.
    https://github.com/MadryLab/cox/blob/master/cox/store.py
    Copyright (c) 2018 Andrew Ilyas, Logan Engstrom, licensed under the MIT license,
    cf. 3rd-party-licenses.txt file in the root directory of this source tree.
    """

    def log_tb(self, table_name, update_dict, summary_type='scalar', step=None):
        """
        Log to only tensorboard.

        Args:
            table_name (str) : which table to log to
            update_dict (dict) : values to log and store as a dictionary of
                column mapping to value.
            summary_type (str) : what type of summary to log to tensorboard as
            step: which step index to insert datapoint
        """

        table = self.tables[table_name]
        update_dict = _clean_dict(update_dict, table.schema)

        tb_func = getattr(self.tensorboard, 'add_%s' % summary_type)
        step = step if step else table.nrows

        for name, value in update_dict.items():
            tb_func('/'.join([table_name, name]), value, step)

    def log_table_and_tb(self, table_name, update_dict, summary_type='scalar', step=None):
        """
        Log to a table and also a tensorboard.

        Args:
            table_name (str) : which table to log to
            update_dict (dict) : values to log and store as a dictionary of
                column mapping to value.
            summary_type (str) : what type of summary to log to tensorboard as
            step: which step index to insert datapoint
        """

        table = self.tables[table_name]
        update_dict = _clean_dict(update_dict, table.schema)

        tb_func = getattr(self.tensorboard, 'add_%s' % summary_type)
        step = step if step else table.nrows

        for name, value in update_dict.items():
            tb_func('/'.join([table_name, name]), value, step)

        table.update_row(update_dict)

    def load(self, table: str, key: str, data_save_type: str, iteration: int = -1, **kwargs):
        """
        Load data from store.
            table: name of table to load from
            key: key of value to load
            data_save_type: Type of the data to be loaded. One of 'object' 'state_dict', or 'pickle'.
            iteration: Iteration checkpoint to load
            kwargs:
        Returns:

        """

        if data_save_type == 'object':
            return self.tables[table].get_object(self.tables[table].df[key].iloc[iteration], **kwargs)
        elif data_save_type == 'state_dict':
            return self.tables[table].get_state_dict(self.tables[table].df[key].iloc[iteration], **kwargs)
        elif data_save_type == 'pickle':
            return self.tables[table].get_pickle(self.tables[table].df[key].iloc[iteration], **kwargs)
        else:
            return self.tables[table].df[key].iloc[iteration]
