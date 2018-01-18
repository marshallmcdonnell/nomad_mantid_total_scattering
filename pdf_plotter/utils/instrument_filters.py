
from traits.api \
    import HasTraits, Str, Float, List

class Filter(HasTraits):

    title = Str
    xmin  = Float
    xmax  = Float

class NomadFilters(HasTraits):
    
    bank1_filter = Filter(title = 'Bank: 8.60',
                          xmin  = 1.0,
                          xmax  = 3.0)

    bank2_filter = Filter(title = 'Bank: 15.10',
                          xmin  = 2.0,
                          xmax  = 7.5)
    
    bank3_filter = Filter(title = 'Bank: 31.00',
                          xmin  = 2.0,
                          xmax  = 10.0)

    bank4_filter = Filter(title = 'Bank: 65.00',
                          xmin  = 2.3,
                          xmax  = 34.0)

    bank5_filter = Filter(title = 'Bank: 120.40',
                          xmin  = 4.5,
                          xmax  = 40.0)


    bank6_filter = Filter(title = 'Bank: 150.10',
                          xmin  = 4.8,
                          xmax  = 40.0)


    list_of_filters = List(Filter, [bank1_filter,
                                    bank2_filter,
                                    bank3_filter,
                                    bank4_filter,
                                    bank5_filter,
                                    bank6_filter])
